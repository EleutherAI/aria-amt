import os
import hashlib
import argparse
import multiprocessing

from pydub import AudioSegment


def hash_audio_file(file_path):
    """Hash the audio content of an MP3 file."""
    try:
        audio = AudioSegment.from_mp3(file_path)
        raw_data = audio.raw_data
    except Exception as e:
        print(e)
        return file_path, -1
    else:
        return file_path, hashlib.sha256(raw_data).hexdigest()


def find_duplicates(root_dir):
    """Find and remove duplicate MP3 files in the directory and its subdirectories."""
    duplicates = []
    mp3_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mp3"):
                mp3_paths.append(os.path.join(root, file))

    with multiprocessing.Pool() as pool:
        hashes = pool.map(hash_audio_file, mp3_paths)

    seen_hash = {}
    for p, h in hashes:
        if seen_hash.get(h, False) is True:
            print("Seen dupe")
            duplicates.append(p)
        else:
            print("Seen orig")
            seen_hash[h] = True

    return duplicates


def remove_duplicates(duplicate_files):
    """Remove the duplicate files."""
    for file in duplicate_files:
        os.remove(file)
        print(f"Removed duplicate file: {file}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate MP3 files based on audio content."
    )
    parser.add_argument(
        "dir", type=str, help="Directory to scan for duplicate MP3 files."
    )
    args = parser.parse_args()

    root_directory = args.dir
    duplicates = find_duplicates(root_directory)

    if duplicates:
        print(f"Found {len(duplicates)} duplicates. Removing...")
        remove_duplicates(duplicates)
    else:
        print("No duplicates found.")


if __name__ == "__main__":
    main()
