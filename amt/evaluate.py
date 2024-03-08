import glob
from tqdm.auto import tqdm
import pretty_midi
import numpy as np
import mir_eval
import json
import os

def midi_to_intervals_and_pitches(midi_file_path):
    """
    This function reads a MIDI file and extracts note intervals and pitches
    suitable for use with mir_eval's transcription evaluation functions.
    """
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Prepare lists to collect note intervals and pitches
    notes = []
    for instrument in midi_data.instruments:
        # Skip drum instruments
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append([note.start, note.end, note.pitch])
    notes = sorted(notes, key=lambda x: x[0])
    notes = np.array(notes)
    intervals, pitches = notes[:, :2], notes[:, 2]
    intervals -= intervals[0][0]
    return intervals, pitches


def midi_to_hz(note, shift=0):
    """
    Convert MIDI to HZ.

    Shift, if != 0, is subtracted from the MIDI note.
        Use "2" for the hFT augmented model transcriptions, else pitches won't match.
    """
    # the one used in hFT transformer
    return 440.0 * (2.0 ** (note.astype(int) - shift - 69) / 12)
    a = 440  # frequency of A (common value is 440Hz)
    # return (a / 32) * (2 ** ((note - 9) / 12))


def evaluate_mir_eval(est_dir, ref_dir, output_stats_file=None, est_shift=0):
    """
    Evaluate the estimated pitches against the reference pitches using mir_eval.
    """
    # Evaluate the estimated pitches against the reference pitches
    ref_midi_files = glob.glob(f"{ref_dir}/*.mid*")
    est_midi_files = glob.glob(f"{est_dir}/*.mid*")

    est_ref_pairs = []
    for est_fpath in est_midi_files:
        ref_fpath = os.path.join(ref_dir, os.path.basename(est_fpath))
        if ref_fpath in ref_midi_files:
            est_ref_pairs.append((est_fpath, ref_fpath))
        if ref_fpath.replace(".mid", ".midi") in ref_midi_files:
            est_ref_pairs.append((est_fpath, ref_fpath.replace(".mid", ".midi")))
        else:
            print(f"Reference file not found for {est_fpath} (ref file: {ref_fpath})")

    output_fhandle = open(output_stats_file, "w") if output_stats_file is not None else None

    for est_file, ref_file in tqdm(est_ref_pairs):
        ref_intervals, ref_pitches = midi_to_intervals_and_pitches(ref_file)
        est_intervals, est_pitches = midi_to_intervals_and_pitches(est_file)
        ref_pitches_hz = midi_to_hz(ref_pitches)
        est_pitches_hz = midi_to_hz(est_pitches, est_shift)
        scores = mir_eval.transcription.evaluate(ref_intervals, ref_pitches_hz, est_intervals, est_pitches_hz)
        if output_fhandle is not None:
            output_fhandle.write(json.dumps(scores))
            output_fhandle.write("\n")
        else:
            print(json.dumps(scores, indent=4))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage="evaluate <command> [<args>]")
    parser.add_argument("--est-dir", type=str, help="Path to the directory containing the transcribed files")
    parser.add_argument("--ref-dir", type=str, help="Path to the directory containing the reference files")
    parser.add_argument(
        '--output-stats-file',
        default=None,
        type=str, help="Path to the file to save the evaluation stats"
    )

    # add mir_eval, mir_eval aug and dtw subparsers
    subparsers = parser.add_subparsers(help="sub-command help")
    mir_eval_parse = subparsers.add_parser("run_mir_eval", help="Run standard mir_eval evaluation on MAESTRO test set.")
    mir_eval_parse.add_argument('--shift', type=int, default=0, help="Shift to apply to the estimated pitches.")

    mir_eval_aug_parse = subparsers.add_parser(
        "run_mir_eval_aug", help="Run mir_eval evaluation on MAESTRO test set with augmented data."
    )
    mir_eval_aug_parse.add_argument('--shift', type=int, default=0, help="Shift to apply to the estimated pitches.")
    dtw_eval_parse = subparsers.add_parser("run_dtw", help="Run dynamic time warping evaluation on a specified dataset.")

    args = parser.parse_args()
    if not hasattr(args, "command"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)

    if args.command == "run_mir_eval":
        evaluate_mir_eval(args.est_dir, args.ref_dir, args.output_stats_file, args.shift)
