import glob
from tqdm.auto import tqdm
import numpy as np
import mir_eval
import json
import os

from aria.data.midi import MidiDict, get_duration_ms


def midi_to_intervals_and_pitches(midi_file_path):
    mid_dict = MidiDict.from_midi(midi_file_path)
    mid_dict.resolve_pedal()

    intervals, pitches, velocities = [], [], []
    for note_msg in mid_dict.note_msgs:
        pitch = note_msg["data"]["pitch"]
        onset_s = (
            get_duration_ms(
                start_tick=0,
                end_tick=note_msg["data"]["start"],
                tempo_msgs=mid_dict.tempo_msgs,
                ticks_per_beat=mid_dict.ticks_per_beat,
            )
            * 1e-3
        )
        offset_s = (
            get_duration_ms(
                start_tick=0,
                end_tick=note_msg["data"]["end"],
                tempo_msgs=mid_dict.tempo_msgs,
                ticks_per_beat=mid_dict.ticks_per_beat,
            )
            * 1e-3
        )
        velocity = note_msg["data"]["velocity"]

        if onset_s >= offset_s:
            print("Skipping duration zero note")
            continue

        intervals.append([onset_s, offset_s])
        pitches.append(pitch)
        velocities.append(velocity)

    return np.array(intervals), np.array(pitches), np.array(velocities)


def midi_to_hz(note, shift=0):
    """
    Convert MIDI to HZ.

    Shift, if != 0, is subtracted from the MIDI note.
        Use "2" for the hFT augmented model transcriptions, else pitches won't match.
    """
    # the one used in hFT transformer
    return 440.0 * (2.0 ** (note.astype(int) - shift - 69) / 12)
    # a = 440  # frequency of A (common value is 440Hz)
    # return (a / 32) * (2 ** ((note - 9) / 12))


def get_matched_files(est_dir: str, ref_dir: str):
    # We assume that the files have the same path relative to their directory

    res = []
    est_paths = glob.glob(os.path.join(est_dir, "**/*.mid"), recursive=True)
    if len(est_paths) == 0:
        est_paths = glob.glob(
            os.path.join(est_dir, "**/*.midi"), recursive=True
        )
    print(f"found {len(est_paths)} est files")

    for est_path in est_paths:
        est_rel_path = os.path.relpath(est_path, est_dir)
        ref_path = os.path.join(
            ref_dir, os.path.splitext(est_rel_path)[0] + ".mid"
        )
        if os.path.isfile(ref_path):
            res.append((est_path, ref_path))
        else:
            ref_path = os.path.join(
                ref_dir, os.path.splitext(est_rel_path)[0] + ".midi"
            )
            if os.path.isfile(ref_path):
                res.append((est_path, ref_path))

    print(f"found {len(res)} matched est-ref pairs")

    return res


def get_matched_files_direct(est_dir: str, ref_dir: str):
    # Helper to extract filenames with normalized extensions
    def get_filenames(paths):
        normalized_files = {}
        for path in paths:
            basename = os.path.basename(path)
            name, ext = os.path.splitext(basename)

            name = name[:-12] if name.endswith("_transcribed") else name

            if ext in [".mid", ".midi"]:
                normalized_files[name] = path
        return normalized_files

    # Gather all potential MIDI files in both directories
    est_files = glob.glob(os.path.join(est_dir, "**/*.*"), recursive=True)
    ref_files = glob.glob(os.path.join(ref_dir, "**/*.*"), recursive=True)

    # Map filenames to their full paths with normalized extensions
    est_file_map = get_filenames(est_files)
    ref_file_map = get_filenames(ref_files)

    # Find matching files by filename disregarding extension differences
    matched_files = []
    for filename, ref_path in ref_file_map.items():
        if filename in est_file_map:
            matched_files.append((est_file_map[filename], ref_path))

    print(f"found {len(est_file_map)} MIDI files in estimation directory")
    print(f"found {len(ref_file_map)} MIDI files in reference directory")
    print(f"found {len(matched_files)} matched MIDI file pairs")

    return matched_files


def get_avg_scores(scores):
    totals = {}
    counts = {}
    for d in scores:
        for key, value in d.items():
            if key == "f_name":
                continue
            totals[key] = totals.get(key, 0) + value
            counts[key] = counts.get(key, 0) + 1
    averages = {f"{key}_avg": totals[key] / counts[key] for key in totals}
    return averages


def evaluate_mir_eval(est_dir, ref_dir, output_stats_file=None, est_shift=0):
    """
    Evaluate the estimated pitches against the reference pitches using mir_eval.
    """

    est_ref_pairs = get_matched_files(est_dir, ref_dir)
    if len(est_ref_pairs) == 0:
        print("Failed to find files, trying direct search")
        est_ref_pairs = get_matched_files_direct(est_dir, ref_dir)

    output_fhandle = (
        open(output_stats_file, "w") if output_stats_file is not None else None
    )

    res = []
    for est_file, ref_file in tqdm(est_ref_pairs):
        ref_intervals, ref_pitches, ref_velocities = (
            midi_to_intervals_and_pitches(ref_file)
        )
        est_intervals, est_pitches, est_velocities = (
            midi_to_intervals_and_pitches(est_file)
        )
        ref_pitches_hz = midi_to_hz(ref_pitches)
        est_pitches_hz = midi_to_hz(est_pitches, est_shift)

        scores = mir_eval.transcription.evaluate(
            ref_intervals,
            ref_pitches_hz,
            est_intervals,
            est_pitches_hz,
        )

        prec_vel, recall_vel, f1_vel, _ = (
            mir_eval.transcription_velocity.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                ref_velocities=ref_velocities,
                est_intervals=est_intervals,
                est_pitches=est_pitches,
                est_velocities=est_velocities,
            )
        )

        scores["Precision_vel"] = prec_vel
        scores["Recall_vel"] = recall_vel
        scores["F1_vel"] = f1_vel
        scores["f_name"] = est_file
        res.append(scores)

    avg_scores = get_avg_scores(res)
    output_fhandle.write(json.dumps(avg_scores))
    output_fhandle.write("\n")

    res.sort(key=lambda x: x["F-measure"])
    for s in res:
        output_fhandle.write(json.dumps(s))
        output_fhandle.write("\n")


def evaluate_single(est_file, ref_file):
    ref_intervals, ref_pitches, ref_velocities = midi_to_intervals_and_pitches(
        ref_file
    )
    est_intervals, est_pitches, est_velocities = midi_to_intervals_and_pitches(
        est_file
    )
    ref_pitches_hz = midi_to_hz(ref_pitches)
    est_pitches_hz = midi_to_hz(est_pitches)

    return mir_eval.transcription.evaluate(
        ref_intervals,
        ref_pitches_hz,
        est_intervals,
        est_pitches_hz,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(usage="evaluate <command> [<args>]")
    parser.add_argument(
        "--est-dir",
        type=str,
        help="Path to the directory containing either the transcribed MIDI files or WAV files to be transcribed.",
    )
    parser.add_argument(
        "--ref-dir",
        type=str,
        help="Path to the directory containing the reference files (we'll use gold MIDI for mir_eval, WAV for dtw).",
    )
    parser.add_argument(
        "--output-stats-file",
        default=None,
        type=str,
        help="Path to the file to save the evaluation stats",
    )

    args = parser.parse_args()
    evaluate_mir_eval(
        args.est_dir,
        args.ref_dir,
        args.output_stats_file,
    )
