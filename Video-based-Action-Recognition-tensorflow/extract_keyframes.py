import argparse
import os
from utils import extract_keyframes


def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    l = extract_keyframes(
        args.video_file_path,
        args.output_dir,
        args.method,
        True,
        args.threshold,
        args.num_top_frames,
        args.len_window,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract key frames from a video.")
    parser.add_argument(
        "--video_file_path",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Output",
        help="Directory to save the output key frames.",
    )
    parser.add_argument(
        "--method",
        choices=["TOP_ORDER", "Threshold", "LOCAL_MAXIMA"],
        default="LOCAL_MAXIMA",
        help="The method to extract key frames: 'TOP_ORDER', 'Threshold', or 'LOCAL_MAXIMA'",
    )
    parser.add_argument(
        "--num_top_frames",
        type=int,
        default=50,
        help="The number of top frames to extract if using the 'TOP_ORDER' method.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Used in the 'Threshold' method to define the minimum relative change between consecutive frames to consider a frame as a keyframe.",
    )
    parser.add_argument(
        "--len_window",
        type=int,
        default=50,
        help="The window size for smoothing frame differences in the 'LOCAL_MAXIMA' method",
    )

    args = parser.parse_args()
    main(args)
