# pip install git+https://github.com/alex2awesome/djitw.git

import argparse
import csv
import librosa
import djitw
import pretty_midi
import scipy
import random
import multiprocessing
import os
import warnings
import functools
import glob
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

# Audio/CQT parameters
FS = 22050.0
NOTE_START = 36
N_NOTES = 48
HOP_LENGTH = 1024

# DTW parameters
GULLY = 0.96


def compute_cqt(audio_data):
    """Compute the CQT and frame times for some audio data"""
    # Compute CQT
    cqt = librosa.cqt(
        audio_data,
        sr=FS,
        fmin=librosa.midi_to_hz(NOTE_START),
        n_bins=N_NOTES,
        hop_length=HOP_LENGTH,
        tuning=0.0,
    )
    # Compute the time of each frame
    times = librosa.frames_to_time(
        np.arange(cqt.shape[1]), sr=FS, hop_length=HOP_LENGTH
    )
    # Compute log-amplitude
    cqt = librosa.amplitude_to_db(cqt, ref=cqt.max())
    # Normalize and return
    return librosa.util.normalize(cqt, norm=2).T, times


# Had to change this to average chunks for large audio files for cpu reasons
def load_and_run_dtw(args):
    def calc_score(_midi_cqt, _audio_cqt):
        # Nearly all high-performing systems used cosine distance
        distance_matrix = scipy.spatial.distance.cdist(
            _midi_cqt, _audio_cqt, "cosine"
        )

        # Get lowest cost path
        p, q, score = djitw.dtw(
            distance_matrix,
            GULLY,  # The gully for all high-performing systems was near 1
            np.median(
                distance_matrix
            ),  # The penalty was also near 1.0*median(distance_matrix)
            inplace=False,
        )
        # Normalize by path length, normalize by distance matrix submatrix within path
        score = score / len(p)
        score = (
            score / distance_matrix[p.min() : p.max(), q.min() : q.max()].mean()
        )

        return score

    audio_file, midi_file = args
    # Load in the audio data
    audio_data, _ = librosa.load(audio_file, sr=FS)
    audio_cqt, audio_times = compute_cqt(audio_data)

    midi_object = pretty_midi.PrettyMIDI(midi_file)
    midi_audio = midi_object.fluidsynth(fs=FS)
    midi_cqt, midi_times = compute_cqt(midi_audio)

    # Truncate to save on compute time for long tracks
    MAX_LEN = 10000
    total_len = midi_cqt.shape[0]
    if total_len > MAX_LEN:
        idx = 0
        scores = []
        while idx < total_len:
            scores.append(
                calc_score(
                    _midi_cqt=midi_cqt[idx : idx + MAX_LEN, :],
                    _audio_cqt=audio_cqt[idx : idx + MAX_LEN, :],
                )
            )
            idx += MAX_LEN

        max_score = max(scores)
        avg_score = sum(scores) / len(scores) if scores else 1.0

    else:
        avg_score = calc_score(_midi_cqt=midi_cqt, _audio_cqt=audio_cqt)
        max_score = avg_score

    return midi_file, avg_score, max_score


# I changed wav with mp3 in here :/
def get_matched_files(audio_dir: str, mid_dir: str):
    # We assume that the files have the same path relative to their directory
    res = []
    wav_paths = glob.glob(
        os.path.join(audio_dir, "**/*.mp[34]"), recursive=True
    )
    print(f"found {len(wav_paths)} mp3 files")

    for wav_path in wav_paths:
        input_rel_path = os.path.relpath(wav_path, audio_dir)
        mid_path = os.path.join(
            mid_dir, os.path.splitext(input_rel_path)[0] + ".mid"
        )
        if os.path.isfile(mid_path):
            res.append((wav_path, mid_path))

    print(f"found {len(res)} matched mp3-midi pairs")

    return res


def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get("timeout", None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)
        return out
    except multiprocessing.TimeoutError:
        return None, None, None
    except Exception as e:
        print(e)
        return None, None, None
    finally:
        p.close()
        p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="amplitude_to_db was called on complex input",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("-audio_dir", help="dir containing .wav files")
    parser.add_argument(
        "-mid_dir", help="dir containing .mid files", default=None
    )
    parser.add_argument(
        "-output_file", help="path to output file", default=None
    )
    args = parser.parse_args()

    matched_files = get_matched_files(
        audio_dir=args.audio_dir, mid_dir=args.mid_dir
    )

    results = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[row["mid_path"]] = {
                    "avg_score": row["avg_score"],
                    "max_score": row["max_score"],
                }

    matched_files = [
        (audio_path, mid_path)
        for audio_path, mid_path in matched_files
        if mid_path not in results.keys()
    ]
    random.shuffle(matched_files)
    print(f"loaded {len(results)} results")
    print(f"calculating scores for {len(matched_files)}")

    score_csv = open(args.output_file, "a")
    csv_writer = csv.writer(score_csv)
    csv_writer.writerow(["mid_path", "avg_score", "max_score"])

    with multiprocessing.Pool() as pool:
        abortable_func = functools.partial(
            abortable_worker, load_and_run_dtw, timeout=15000
        )
        scores = pool.imap_unordered(abortable_func, matched_files)

        skipped = 0
        processed = 0
        for mid_path, avg_score, max_score in scores:
            if avg_score is not None and max_score is not None:
                csv_writer.writerow([mid_path, avg_score, max_score])
                score_csv.flush()
            else:
                print(f"timeout")
                skipped += 1

            processed += 1
            if processed % 10 == 0:
                print(f"PROCESSED: {processed}/{len(matched_files)}")
                print(f"***")

        print(f"skipped: {skipped}")
