import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Collect evaluation scores for one checkpoint.")
    parser.add_argument("-mn", "--model_name", type=str, default="example_ac4_4%")
    parser.add_argument("-m", "--mode", type=str, default="AC3", choices=["AC3", "AC4"])
    parser.add_argument("--model_id", type=str, default="model-200000")
    parser.add_argument("--inference_dir", type=str, default="./inference")
    parser.add_argument("--output_name", type=str, default="result_post.txt")
    return parser.parse_args()


def read_score_lines(score_path):
    if not os.path.isfile(score_path):
        raise FileNotFoundError(f"Score file not found: {score_path}")

    with open(score_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    folder_path = os.path.join(args.inference_dir, args.model_name, args.mode)
    affs_folder = "affs_" + args.model_id
    score_path = os.path.join(folder_path, affs_folder, "scores_post.txt")
    output_path = os.path.join(folder_path, args.output_name)

    lines = read_score_lines(score_path)
    with open(output_path, "w") as result_file:
        result_file.write(f"{affs_folder}\n")
        for line in lines:
            result_file.write(line + "\n")

    print(f"Results have been written to {output_path}.")


if __name__ == "__main__":
    main()
