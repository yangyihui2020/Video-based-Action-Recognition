import os
import argparse
import time
import cv2
import operator
import numpy as np
from scipy.signal import argrelextrema
from moviepy.video.io.VideoFileClip import VideoFileClip


def split_video(input_video_path, output_dir, output_video_name, clip_duration):

    video = VideoFileClip(input_video_path)
    video_duration = video.duration  # 获取视频总时长

    # 计算视频的开始时间和结束时间
    start_time = 0
    clip_number = 1

    while start_time < video_duration:
        # 计算片段的结束时间，防止超出视频总时长
        end_time = min(start_time + clip_duration, video_duration)

        # 提取片段并保存为新文件
        clip = video.subclip(start_time, end_time)
        clip_video_name = (
            f"{os.path.splitext(output_video_name)[0]}_part_{clip_number}.mp4"
        )
        output_path = os.path.join(output_dir, clip_video_name)
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"Saved clip {clip_number}: {start_time} to {end_time} seconds.")

        # 更新起始时间和片段编号
        start_time = end_time
        clip_number += 1


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


def smooth(x, window_len=13, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal

    example:
    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    """
    # print(len(x), window_len)
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    # print(len(s))

    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1 : -window_len + 1]


class Frame:
    """
    class to hold information about each frame

    """

    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(a, b):
    x = (b - a) / max(a, b)
    print(x)
    return x


def get_video_info(video_path):
    """
    This function gets information about a video file, including its resolution, FPS, and total number of frames.

    Parameters:
    video_path (str): The path to the video file.

    Returns:
    List[int]: A list containing the video resolution, FPS, and total number of frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: fail to open video file")
        exit()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"video resolution: {int(width)}x{int(height)}, FPS: {fps}, total frames: {total_frames}")
    cap.release()
    return width, height, fps, total_frames


def main(args):
    method = args.method
    THRESH = args.threshold
    NUM_TOP_FRAMES = args.num_top_frames
    videopath = args.video_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, args.output_video_name)
    len_window = args.len_window
    frame_rate = args.frame_rate
    
    width, height, fps, total_frames = get_video_info(videopath)
    if frame_rate == 0:
        frame_rate = fps
        
    print(
        f"target video : {videopath}, video resolution: {width}x{height}, FPS: {fps}, total frames: {total_frames}."
    )
    print("video save directory: " + output_video_path)

    # load video and compute diff between frames
    cap = cv2.VideoCapture(videopath)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 指定视频编码格式为mp4
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    timer = Timer()
    i = 0

    timer.start()
    while success:
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame

        i = i + 1
        success, frame = cap.read()
    cap.release()

    # compute keyframe
    keyframe_id_set = set()
    if method == "TOP_ORDER":
        print("Using Top Order")
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_set.add(keyframe.id)
    if method == "Threshold":
        print("Using Threshold")
        for i in range(1, len(frames)):
            if (
                rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff))
                >= THRESH
            ):
                keyframe_id_set.add(frames[i].id)
    if method == "LOCAL_MAXIMA":
        print("Using Local Maxima")
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)

    print(f"total {len(keyframe_id_set)} / {total_frames} key frames")
    """
    plt.figure(figsize=(40, 20))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=100))
    plt.stem(sm_diff_array)
    plt.savefig(dir + "plot.png")
    """

    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    # keyframes = []
    success, frame = cap.read()
    idx = 0
    while success:
        if idx in keyframe_id_set:
            video.write(frame)
            keyframe_id_set.remove(idx)
        idx = idx + 1
        success, frame = cap.read()
    cap.release()
    video.release()
    print(f"Video {output_video_path} has been created successfully.")
    
    if args.split:
        print("Splitting video into smaller parts...")
        split_video(output_video_path, output_dir, args.output_video_name, args.clip_duration)
        print("Splitting completed.")

    print(f"Processing time：{timer.stop()} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict on a video using a pre-trained model."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the input video."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Output",
        help="Directory to save the output video.",
    )
    parser.add_argument(
        "--output_video_name",
        type=str,
        default="output.mp4",
        help="Name of the output video file.",
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=0,
        help="Fps of the output video.",
    )
    parser.add_argument(
        "--method",
        choices=["TOP_ORDER", "Threshold", "LOCAL_MAXIMA"],
        default="LOCAL_MAXIMA",
        help="The method to extract key frames: 'TOP_ORDER','Threshold', or 'LOCAL_MAXIMA'",
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
    parser.add_argument(
        "--split", action="store_true", help="Whether to split the output video."
    )
    parser.add_argument(
        "--clip_duration",
        type=int,
        default=60,
        help="Duration of each video clip (seconds)",
    )

    args = parser.parse_args()
    main(args)
