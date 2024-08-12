import argparse
import csv
import os
import shutil


# Calculate percentiles without using numpy
def calculate_percentiles(data, percentiles):
    data_sorted = sorted(data)
    n = len(data_sorted)
    results = []
    for percentile in percentiles:
        k = (n - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f + 1 < n:
            result = data_sorted[f] + c * (data_sorted[f + 1] - data_sorted[f])
        else:
            result = data_sorted[f]
        results.append(result)
    return results


def main(mid_dir, output_dir, score_file, max_score, dry):
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    scores = {}
    with open(score_file, "r") as f:
        reader = csv.DictReader(f)
        failures = 0
        for row in reader:
            try:
                if 0.0 < float(row["avg_score"]) < 1.0:
                    scores[row["mid_path"]] = float(row["avg_score"])
            except Exception as e:
                pass

    print(f"{failures} failures")
    print(f"found {len(scores.items())} mid-score pairs")

    print("top 50 by score:")
    for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
        :50
    ]:
        print(f"{v}: {k}")
    print("bottom 50 by score:")
    for k, v in sorted(scores.items(), key=lambda item: item[1])[:50]:
        print(f"{v}: {k}")

    # Define the percentiles to calculate
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    floats = [v for k, v in scores.items()]

    # Calculate the percentiles
    print(f"percentiles: {calculate_percentiles(floats, percentiles)}")

    cnt = 0
    for mid_path, score in scores.items():
        mid_rel_path = os.path.relpath(mid_path, mid_dir)
        output_path = os.path.join(output_dir, mid_rel_path)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        if score < max_score:
            if args.dry is not True:
                shutil.copyfile(mid_path, output_path)
        else:
            cnt += 1

    print(f"excluded {cnt}/{len(scores.items())} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mid_dir", help="dir containing .mid files", default=None
    )
    parser.add_argument(
        "-output_dir", help="dir containing .mid files", default=None
    )
    parser.add_argument("-score_file", help="path to output file", default=None)
    parser.add_argument(
        "-max_score", type=float, help="path to output file", default=None
    )
    parser.add_argument("-dry", action="store_true", help="path to output file")
    args = parser.parse_args()

    main(
        args.mid_dir, args.output_dir, args.score_file, args.max_score, args.dry
    )
