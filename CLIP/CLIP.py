import tkinter as tk
from tkinter import filedialog
from moviepy.editor import VideoFileClip

import sys
import os

# 重定向标准输出和错误输出到空设备，这样就不会在控制台打印输出
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# 定义一个函数来打开文件对话框并返回视频文件路径
def open_video_file():
    file_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        video_path_entry.delete(0, tk.END)
        video_path_entry.insert(0, file_path)
        status_label.config(text="视频文件已选择，文件路径为：" + file_path, fg="black")


# 定义一个函数来保存视频
def save_video():
    file_path = filedialog.asksaveasfilename(title="保存视频文件", defaultextension=".mp4",
                                             filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        output_file_entry.delete(0, tk.END)
        output_file_entry.insert(0, file_path)
        status_label.config(text="输出文件路径已选择，文件路径为：" + file_path, fg="black")


# 定义一个函数来将时分秒格式的时间转换为秒数
def time_to_seconds(hours, minutes, seconds):
    return hours * 3600 + minutes * 60 + seconds


# 定义一个函数来截取视频
def cut_video():
    try:
        start_hours = int(start_hours_entry.get())
        start_minutes = int(start_minutes_entry.get())
        start_seconds = int(start_seconds_entry.get())
        end_hours = int(end_hours_entry.get())
        end_minutes = int(end_minutes_entry.get())
        end_seconds = int(end_seconds_entry.get())

        start_time = time_to_seconds(start_hours, start_minutes, start_seconds)
        end_time = time_to_seconds(end_hours, end_minutes, end_seconds)

        if start_time >= end_time:
            raise ValueError("起始时间必须小于结束时间")

        clip = VideoFileClip(video_path_entry.get())
        cut_clip = clip.subclip(start_time, end_time)
        status_label.config(text="视频切割中.....", fg="black")
        cut_clip.write_videofile(output_file_entry.get(), codec="libx264")
        status_label.config(text="视频切割完成，输出的视频为：" + output_file_entry.get(), fg="black")
    except ValueError as e:
        status_label.config(text=e, fg="red")
    except Exception as e:
        status_label.config(text="视频处理中发生错误：" + str(e), fg="red")


# 创建主窗口
root = tk.Tk()
root.title("视频截取工具")
root.geometry("950x300")

# 创建一个标签用于显示状态信息，并放置在界面的最上方
status_label = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.grid(row=5, column=0, columnspan=4, padx=5, pady=5,sticky='EW')

# 创建一个标签和输入框来输入视频路径
video_path_label = tk.Label(root, text="⬅这是你选择的输入视频文件路径")
video_path_label.grid(row=0, column=4, padx=5, pady=5)
video_path_entry = tk.Entry(root, width=40)
video_path_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=5)
video_path_button = tk.Button(root, text="选择视频文件", command=open_video_file)
video_path_button.grid(row=0, column=0, padx=5, pady=5)

# 创建输入框来输入起始时间
start_time_label = tk.Label(root, text="起始时间点 例如你想要从00:01:45开始截取，只需要依次输入00,01,45即可")
start_time_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
start_hours_entry = tk.Entry(root, width=5)
start_hours_entry.grid(row=1, column=1, padx=5, pady=5)
start_minutes_entry = tk.Entry(root, width=5)
start_minutes_entry.grid(row=1, column=2, padx=5, pady=5)
start_seconds_entry = tk.Entry(root, width=5)
start_seconds_entry.grid(row=1, column=3, padx=5, pady=5)

# 创建输入框来输入结束时间
end_time_label = tk.Label(root, text="结束时间点 例如你想要截取到00:01:49，只需要依次输入00,01,49即可:")
end_time_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
end_hours_entry = tk.Entry(root, width=5)
end_hours_entry.grid(row=2, column=1, padx=5, pady=5)
end_minutes_entry = tk.Entry(root, width=5)
end_minutes_entry.grid(row=2, column=2, padx=5, pady=5)
end_seconds_entry = tk.Entry(root, width=5)
end_seconds_entry.grid(row=2, column=3, padx=5, pady=5)

# 创建输入框来输入输出文件路径和文件名
output_file_label = tk.Label(root, text="⬅这是你选择的输出文件路径")
output_file_label.grid(row=3, column=4, padx=5, pady=5)
output_file_entry = tk.Entry(root, width=40)
output_file_entry.grid(row=3, column=1, columnspan=3, padx=5, pady=5)
output_file_button = tk.Button(root, text="选择输出文件路径", command=save_video)
output_file_button.grid(row=3, column=0, padx=5, pady=5)

# 创建按钮来截取视频
cut_button = tk.Button(root, text="开始截取视频", command=cut_video)
cut_button.grid(row=4, column=0, columnspan=5, sticky="ew", padx=5, pady=10)

# 运行主循环
root.mainloop()