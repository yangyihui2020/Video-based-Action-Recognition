import torch
import cv2
import numpy as np
from moviepy import VideoFileClip
import argparse
import os
from lstm_model import LSTMModel
import torch.nn.functional as F
import time 
from tqdm import tqdm



def predict_on_video_test(video_file_path, output_dir, output_video_name, SEQUENCE_LENGTH, model, device, CLASSES_LIST,actionlist_to_recognise IMAGE_HEIGHT=64, IMAGE_WIDTH=64, confidence_threshold=0.99):
    video_reader = cv2.VideoCapture(video_file_path)
    output_file_path = os.path.join(output_dir, output_video_name)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                  int(fps), (original_video_width, original_video_height))

    times = {}
    times['read_frame']=0
    times['resize_frame']=0
    times['normalize_frame']=0
    times['model_inference']=0
    times['write_frame']=0
    frame_list=[]
    new_frame_list=[]
    frame_index = 0  # 初始化帧索引

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        if not frame.size:
            continue
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame / 255.0
        new_frame_list.append(normalized_frame)
        frame_list.append(frame)

        # 计算每一帧的时间戳（秒）
        timestamp = frame_index / fps
        minutes, seconds = divmod(timestamp, 60)
        # print(f"Frame {frame_index}: Time {int(minutes):02d}:{int(seconds):05.2f}")

        frame_index += 1  # 更新帧索引

    begin_index=-1
    end_index=SEQUENCE_LENGTH
    current_label=''
    if len(frame_list) >= SEQUENCE_LENGTH:
        for i in tqdm(range(0, len(frame_list) - SEQUENCE_LENGTH + 1, 1), desc='Processing sequences'):
            sequence_frames = new_frame_list[i:i + SEQUENCE_LENGTH]
            input_list=[]
            input_list.append(sequence_frames)
            model = model.to(device)
            input_list = np.array(input_list)
            predicted_labels_probabilities = model(torch.from_numpy(input_list).float().to(device))
            probability_tensor = F.softmax(predicted_labels_probabilities, dim=1)
            probability_array = probability_tensor.detach().cpu().numpy()
            probability = probability_array[0]
            predicted_label = np.argmax(probability)
            predicted_confidence = probability[predicted_label]
            frame = frame_list[i]
            if predicted_confidence >= confidence_threshold:
                predicted_class_name = CLASSES_LIST[predicted_label]
                if begin_index==-1:
                    current_label=str(predicted_class_name)
                    begin_index=i
                    end_index=i+SEQUENCE_LENGTH-1
                else:
                    end_index+=1
                start_time = i / fps
                end_time = (i + SEQUENCE_LENGTH) / fps
                
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                if begin_index!=-1:
                    start_time = begin_index / fps
                    if end_index >=len(frame_list):
                        end_index=len(frame_list)-1
                    end_time = (end_index) / fps
                    clip = VideoFileClip(video_file_path)
                    cut_clip = clip.subclipped(start_time, end_time)
                    if current_label in actionlist_to_recognise:
                        cut_clip.write_videofile(f"{output_dir}/{current_label}_{start_time}_.mp4", codec="libx264")
                    begin_index=-1
                    end_index=SEQUENCE_LENGTH
                    current_label=''
            video_writer.write(frame)
    video_reader.release()
    video_writer.release()
import time
import shutil

def main(args):
    # 设置设备
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model=LSTMModel(dataset_dir='',classes_list=args.classes_list,image_height=args.image_height,image_width=args.image_width,sequence_length=args.sequence_length)

    model = lstm_model.create_LRCN_model()
    print("Model created successfully!")
    model_path = args.model_path
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    input_video_file_path =args.input_video_file_path
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    output_video_name = args.output_video_name
    output_video_file_path = os.path.join(output_dir, output_video_name)
    predict_on_video_test(input_video_file_path, output_dir, output_video_name, args.sequence_length, model,device, args.classes_list,args.actionlist_to_recognise,
                          args.image_height,args.image_width,args.confidence_threshold)
    end_time=time.time()
    total_training_time = end_time - start_time
    print(f"Total inference time: {total_training_time:.2f} seconds")
    processed_video = VideoFileClip(output_video_file_path, audio=False, target_resolution=(300, None))
    os.remove(output_video_file_path)
    processed_video.write_videofile(output_video_file_path, codec='libx264', audio=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on a video using a pre-trained model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model.')
    parser.add_argument('--input_video_file_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='./Output', help='Directory to save the output video.')
    parser.add_argument('--output_video_name', type=str, default='output.mp4', help='Name of the output video file.')
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of the sequence.')
    parser.add_argument('--image_height', type=int, default=64, help='Height of the images.')
    parser.add_argument('--image_width', type=int, default=64, help='Width of the images.')
    parser.add_argument('--classes_list', type=str, nargs='+', required=True, help='List of class names.')
    parser.add_argument('--target_resolution', type=int, default=300, help='Target resolution for the video.')
    parser.add_argument('--confidence_threshold', type=float, default=0.99, help='Confidence rate')
    parser.add_argument('--confidence_threshold', type=float, default=0.99, help='Confidence rate')
    parser.add_argument('--actionlist_to_recognise', type=str, nargs='+', required=True, help='actionlist to recognise')

    args = parser.parse_args()
    main(args)
