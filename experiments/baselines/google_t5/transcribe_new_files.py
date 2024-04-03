# ! git clone --branch=main https://github.com/magenta/mt3
# !python3 -m pip install jax[cuda11_local] nest-asyncio -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# copy checkpoints
# ! gsutil -q -m cp -r gs://mt3/checkpoints .

import numpy as np
import tensorflow.compat.v2 as tf
import functools
import gin
import jax
import note_seq
import seqio
import t5
import t5x
import librosa

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies
from scipy.io import wavfile
import os
import glob
here = os.path.dirname(__file__)

def download_model():
    # download model
    import gdown
    import tarfile
    url = "https://drive.google.com/file/d/1H9i8AszhJf9xonSaY6YkVoCkh1KHxYSk/view?usp=sharing"
    output = os.path.join(here, "model_files/checkpoint.tar.gz")
    model_files_dirname = os.path.dirname(output)
    if not os.path.exists(model_files_dirname):
        os.makedirs(model_files_dirname)
    gdown.download(url, output, fuzzy=True)
    tar = tarfile.open(output)
    tar.extractall(model_files_dirname)
    tar.close()
    # download configs
    import wget
    gin_dir = os.path.join(here, 'model_files', 'gin')
    if not os.path.exists(gin_dir):
        os.makedirs(gin_dir)
    url_1 = 'https://raw.githubusercontent.com/magenta/mt3/main/mt3/gin/model.gin'
    url_2 = 'https://raw.githubusercontent.com/magenta/mt3/main/mt3/gin/ismir2021.gin'
    url_3 = 'https://raw.githubusercontent.com/magenta/mt3/main/mt3/gin/mt3.gin'
    wget.download(url_1, os.path.join(gin_dir, 'model.gin'))
    wget.download(url_2, os.path.join(gin_dir, 'ismir2021.gin'))
    wget.download(url_3, os.path.join(gin_dir, 'mt3.gin'))


class InferenceModel(object):
  """Wrapper of T5X model for music transcription."""

  def __init__(self, checkpoint_path, model_type='mt3'):

    # Model Constants.
    if model_type == 'ismir2021':
      num_velocity_bins = 127
      self.encoding_spec = note_sequences.NoteEncodingSpec
      self.inputs_length = 512
    elif model_type == 'mt3':
      num_velocity_bins = 1
      self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
      self.inputs_length = 256
    else:
      raise ValueError('unknown model_type: %s' % model_type)

    gin_files = [
        os.path.join(here, 'model_files', 'gin', 'model.gin'),
        os.path.join(here, 'model_files', 'gin', f'{model_type}.gin')
    ]

    self.batch_size = 8
    self.outputs_length = 1024
    self.sequence_length = {
        'inputs': self.inputs_length,
        'targets': self.outputs_length
    }

    self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

    # Build Codecs and Vocabularies.
    self.spectrogram_config = spectrograms.SpectrogramConfig()
    self.codec = vocabularies.build_codec(
        vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins)
    )
    self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
    self.output_features = {
        'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
        'targets': seqio.Feature(vocabulary=self.vocabulary),
    }

    # Create a T5X model.
    self._parse_gin(gin_files)
    self.model = self._load_model()

    # Restore from checkpoint.
    self.restore_from_checkpoint(checkpoint_path)

  @property
  def input_shapes(self):
    return {
          'encoder_input_tokens': (self.batch_size, self.inputs_length),
          'decoder_input_tokens': (self.batch_size, self.outputs_length)
    }

  def _parse_gin(self, gin_files):
    """Parse gin files used to train the model."""
    gin_bindings = [
        'from __gin__ import dynamic_registration',
        'from mt3 import vocabularies',
        'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
        'vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS'
    ]
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          gin_files, gin_bindings, finalize_config=False)

  def _load_model(self):
    """Load up a T5X `Model` after parsing training gin config."""
    model_config = gin.get_configurable(network.T5Config)()
    module = network.Transformer(config=model_config)
    return models.ContinuousInputsEncoderDecoderModel(
        module=module,
        input_vocabulary=self.output_features['inputs'].vocabulary,
        output_vocabulary=self.output_features['targets'].vocabulary,
        optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
        input_depth=spectrograms.input_depth(self.spectrogram_config))


  def restore_from_checkpoint(self, checkpoint_path):
    """Restore training state from checkpoint, resets self._predict_fn()."""
    train_state_initializer = t5x.utils.TrainStateInitializer(
      optimizer_def=self.model.optimizer_def,
      init_fn=self.model.get_initial_variables,
      input_shapes=self.input_shapes,
      partitioner=self.partitioner)

    restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
        path=checkpoint_path, mode='specific', dtype='float32')

    train_state_axes = train_state_initializer.train_state_axes
    self._predict_fn = self._get_predict_fn(train_state_axes)
    self._train_state = train_state_initializer.from_checkpoint_or_scratch(
        [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))

  @functools.lru_cache()
  def _get_predict_fn(self, train_state_axes):
    """Generate a partitioned prediction function for decoding."""
    def partial_predict_fn(params, batch, decode_rng):
      return self.model.predict_batch_with_aux(
          params, batch, decoder_params={'decode_rng': None})
    return self.partitioner.partition(
        partial_predict_fn,
        in_axis_resources=(
            train_state_axes.params,
            t5x.partitioning.PartitionSpec('data',), None),
        out_axis_resources=t5x.partitioning.PartitionSpec('data',)
    )

  def predict_tokens(self, batch, seed=0):
    """Predict tokens from preprocessed dataset batch."""
    prediction, _ = self._predict_fn(
        self._train_state.params, batch, jax.random.PRNGKey(seed))
    return self.vocabulary.decode_tf(prediction).numpy()

  def __call__(self, audio):
    """Infer note sequence from audio samples.
    Args:
      audio: 1-d numpy array of audio samples (16kHz) for a single example.
    Returns:
      A note_sequence of the transcribed audio.
    """
    ds = self.audio_to_dataset(audio)
    ds = self.preprocess(ds)

    model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
        ds, task_feature_lengths=self.sequence_length)
    model_ds = model_ds.batch(self.batch_size)

    inferences = (tokens for batch in model_ds.as_numpy_iterator()
                  for tokens in self.predict_tokens(batch))

    predictions = []
    for example, tokens in zip(ds.as_numpy_iterator(), inferences):
      predictions.append(self.postprocess(tokens, example))

    result = metrics_utils.event_predictions_to_ns(
        predictions, codec=self.codec, encoding_spec=self.encoding_spec)
    return result['est_ns']

  def audio_to_dataset(self, audio):
    """Create a TF Dataset of spectrograms from input audio."""
    frames, frame_times = self._audio_to_frames(audio)
    return tf.data.Dataset.from_tensors({
        'inputs': frames,
        'input_times': frame_times,
    })

  def _audio_to_frames(self, audio):
    """Compute spectrogram frames from audio."""
    frame_size = self.spectrogram_config.hop_width
    padding = [0, frame_size - len(audio) % frame_size]
    audio = np.pad(audio, padding, mode='constant')
    frames = spectrograms.split_audio(audio, self.spectrogram_config)
    num_frames = len(audio) // frame_size
    times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
    return frames, times

  def preprocess(self, ds):
    pp_chain = [
        functools.partial(
            t5.data.preprocessors.split_tokens_to_inputs_length,
            sequence_length=self.sequence_length,
            output_features=self.output_features,
            feature_key='inputs',
            additional_feature_keys=['input_times']),
        # Cache occurs here during training.
        preprocessors.add_dummy_targets,
        functools.partial(
            preprocessors.compute_spectrograms,
            spectrogram_config=self.spectrogram_config)
    ]
    for pp in pp_chain:
      ds = pp(ds)
    return ds

  def postprocess(self, tokens, example):
    tokens = self._trim_eos(tokens)
    start_time = example['input_times'][0]
    # Round down to nearest symbolic token step.
    start_time -= start_time % (1 / self.codec.steps_per_second)
    return {
        'est_tokens': tokens,
        'start_time': start_time,
        # Internal MT3 code expects raw inputs, not used here.
        'raw_inputs': []
    }

  @staticmethod
  def _trim_eos(tokens):
    tokens = np.array(tokens, np.int32)
    if vocabularies.DECODED_EOS_ID in tokens:
      tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
    return tokens


def load_audio(data, sample_rate=None):
    # read in wave data and convert to samples
    f, sr = librosa.load(data, sr=sample_rate)
    return f
    # return note_seq.audio_io.wav_data_to_samples_librosa(data, sample_rate=sample_rate)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # necessary arguments
    parser.add_argument('-input_dir_to_transcribe', default=None, help='file list')
    parser.add_argument('-input_file_to_transcribe', default=None, help='one file')
    parser.add_argument('-output_dir', help='output directory')
    parser.add_argument('-output_file', default=None, help='output file')
    parser.add_argument('-f_config', help='config json file', default=None)
    parser.add_argument('-model_file', help='input model file', default="ismir2021")
    parser.add_argument('-start_index', help='start index', type=int, default=None)
    parser.add_argument('-end_index', help='end index', type=int, default=None)
    parser.add_argument('-sample_rate', help='sample rate', type=int, default=16000)

    args = parser.parse_args()
    MODEL = args.model_file
    checkpoint_path = os.path.join(here, 'model_files', 'checkpoints', MODEL)
    if not os.path.exists(checkpoint_path):
        download_model()
    inference_model = InferenceModel(checkpoint_path, MODEL)
    if args.input_file_to_transcribe is not None:
        files_to_transcribe = [args.input_file_to_transcribe]
    else:
        files_to_transcribe = glob.glob(os.path.join(args.input_dir_to_transcribe, '*.mp3'))
        files_to_transcribe = files_to_transcribe[args.begin_index:args.end_index]
    if args.output_file is not None:
        args.output_file = os.path.join(here, args.output_file)
        midi_output_names = [args.output_file]
        args.output_dir = os.path.dirname(args.output_file)
    else:
        midi_output_names = list(map(os.path.basename, files_to_transcribe))
        midi_output_names = list(map(lambda x: x.replace('.mp3', '.midi'), midi_output_names))
        midi_output_names = list(map(lambda x: os.path.join(args.output_dir, x), midi_output_names))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for n, (mp3_path, midi_path) in enumerate(zip(files_to_transcribe, midi_output_names)):
        print(n, mp3_path)
        if os.path.exists(midi_path):
            continue
        audio = load_audio(mp3_path, sample_rate=args.sample_rate)
        est_ns = inference_model(audio)
        note_seq.sequence_proto_to_midi_file(est_ns, midi_path)


"""
# test one transcription
python transcribe_new_files.py \
    -input_file_to_transcribe ../../../maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_13_01_2004_01-05_ORIG_MID--AUDIO_13_R1_2004_12_Track12_wav.wav \
    -output_file test-output-file.midi

# test multiple transcriptions 
python transcribe_new_files.py \
    -input_dir_to_transcribe /mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/glenn-gould-bach-data \
    -output_dir /mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/glenn-gould-bach-data/kong-model
"""