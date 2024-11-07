# -*- coding: utf-8 -*-
import time
import argparse
import os
import cv2
import operator
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
from scipy.signal import argrelextrema


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

    TODO: the window parameter could be the window itself if an array instead of a string
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


def extract_keyframes(
        video_file_path,
        output_dir="./output",
        method="LOCAL_MAXIMA",
        save=False,
        threshold=0.6,
        num_top_frames=50,
        len_window=50,
    ):
    """
    Extract keyframes from a video using different methods based on frame differences.
    
    Parameters:
    method : str, optional (default="LOCAL_MAXIMA")
        Method used to extract keyframes. Available methods:
        - "TOP_ORDER": Selects the top `num_top_frames` frames based on the highest frame differences.
        - "Threshold": Selects frames where the relative change in frame difference exceeds the `threshold`.
        - "LOCAL_MAXIMA": Selects local maxima from the smoothed frame difference array.
    
    save : bool, optional (default=True)
        If True, the keyframes will be saved as images in the output directory.
    
    threshold : float, optional (default=0.6)
        Used in the "Threshold" method to define the minimum relative change between consecutive frames to consider a frame as a keyframe.
    
    num_top_frames : int, optional (default=50)
        The number of top frames to extract if using the "TOP_ORDER" method.
    
    len_window : int, optional (default=50)
        The window size for smoothing frame differences in the "LOCAL_MAXIMA" method.

    Returns:
    keyframes : list
        A list of extracted keyframes (as frames).
    """
    method_list = ["TOP_ORDER", "Threshold", "LOCAL_MAXIMA"]
    assert method in method_list, f"argument method must in {method_list}"
    cap = cv2.VideoCapture(video_file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    i = 0
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
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:num_top_frames]:
            keyframe_id_set.add(keyframe.id)
    if method == "Threshold":
        for i in range(1, len(frames)):
            if (
                rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff))
                >= threshold
            ):
                keyframe_id_set.add(frames[i].id)
    if method == "LOCAL_MAXIMA":
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)

        """
        plt.figure(figsize=(40, 20))
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=100))
        plt.stem(sm_diff_array)
        plt.savefig(os.path.join(output_dir, "plot.jpg"))
        """
        
    # save all keyframes as image if necessary
    cap = cv2.VideoCapture(video_file_path)
    curr_frame = None
    keyframes = []
    success, frame = cap.read()
    idx = 0
    while success:
        if idx in keyframe_id_set:
            if save:
                name = "keyframe_" + str(idx) + ".jpg"
                cv2.imwrite(os.path.join(output_dir, name), frame)
            keyframes.append(frame)
            keyframe_id_set.remove(idx)
        idx = idx + 1
        success, frame = cap.read()
    cap.release()
    print(f"Successfully extract {len(keyframes)} / {total_frames} key frames.")
    return keyframes