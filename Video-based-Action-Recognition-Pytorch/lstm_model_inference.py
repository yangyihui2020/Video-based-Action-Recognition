import torch
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import argparse
import os
from collections import deque
from lstm_model import LSTMModel
import torch.nn.functional as F
import time 
from tqdm import tqdm

def predict_on_video0(video_file_path, output_file_path, SEQUENCE_LENGTH, model, device,CLASSES_LIST,IMAGE_HEIGHT=64, IMAGE_WIDTH = 64, confidence_threshold=0.99):
    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                  video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    predicted_class_name = ''
    model=model.to(device)
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        if not frame.size:
            continue
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            frame_list=list(frames_queue)
            input_list=[]
            input_list.append(frame_list)
            # input_tensor = torch.tensor(input_list).to(device).float().to(device) #此处被优化
        
            input_list=np.array(input_list)
            input_tensor=torch.from_numpy(input_list).float().to(device)
            predicted_labels_probabilities = model(input_tensor)
            #print(predicted_labels_probabilities)
            probability_tensor = F.softmax(predicted_labels_probabilities, dim=1)

            # 将 PyTorch 张量转换为 NumPy 数组
            probability_array = probability_tensor.detach().cpu().numpy()
            # print(probability_array)

            probability=probability_array[0]
            print(probability)
            predicted_label = np.argmax(probability)
            predicted_confidence = probability[predicted_label]
            if predicted_confidence >= confidence_threshold:
                predicted_class_name = CLASSES_LIST[predicted_label]
                # 在帧上绘制预测类别
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # 将帧写入视频文件
        video_writer.write(frame)
    video_reader.release()
    video_writer.release()


def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH, model, device, CLASSES_LIST, IMAGE_HEIGHT=64, IMAGE_WIDTH=64, confidence_threshold=0.99):
    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                  video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    start_model_time = time.time()
    times = {}
    times['read_frame']=0
    times['resize_frame']=0
    times['normalize_frame']=0
    times['model_inference']=0
    times['write_frame']=0
    frame_list=[]
    new_frame_list=[]
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
    # if len(frame_list) >= SEQUENCE_LENGTH:
    # # 使用数组下标遍历列表，步长为SEQUENCE_LENGTH
    #     for i in range(0, len(frame_list) - SEQUENCE_LENGTH + 1, 1):
    #         print(i)
    #         sequence_frames = new_frame_list[i:i + SEQUENCE_LENGTH]
    #         input_list=[]
    #         input_list.append(sequence_frames)
    #         list.append(input_list)
    
    
    # # input_tensor = torch.tensor(list).float()
    # # input_tensor = torch.tensor(list, device=device).float()
    end_model_time = time.time()
    print(end_model_time - start_model_time)
    if len(frame_list) >= SEQUENCE_LENGTH:
    # 使用数组下标遍历列表，步长为SEQUENCE_LENGTH
        for i in tqdm(range(0, len(frame_list) - SEQUENCE_LENGTH + 1, 1), desc='Processing sequences'):
            # print(i)
            sequence_frames = new_frame_list[i:i + SEQUENCE_LENGTH]
            input_list=[]
            input_list.append(sequence_frames)
            # start_model_time = time.time()
            # input_tensor = torch.tensor(input_list).to(device).float().to(device)
            # end_model_time = time.time()
            # times['model_inference'] = times['model_inference'] + (end_model_time - start_model_time)
            # print(input_tensor.shape)
            model=model.to(device)
            # predicted_labels_probabilities = model(input_tensor)
            input_list=np.array(input_list)
            # print(input_list.shape)
            predicted_labels_probabilities = model(torch.from_numpy(input_list).float().to(device))
            #print(predicted_labels_probabilities)
            probability_tensor = F.softmax(predicted_labels_probabilities, dim=1)

            # 将 PyTorch 张量转换为 CPU 上的张量，然后转换为 NumPy 数组
            probability_array = probability_tensor.detach().cpu().numpy()
            # print(probability_array)

            probability=probability_array[0]
            print(probability)
            predicted_label = np.argmax(probability)
            predicted_confidence = probability[predicted_label]
            frame=frame_list[i]
            if predicted_confidence >= confidence_threshold:
                predicted_class_name = CLASSES_LIST[predicted_label]
                # 在帧上绘制预测类别
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 将帧写入视频文件
        
            video_writer.write(frame)
    video_reader.release()
    video_writer.release()

        
    # while video_reader.isOpened():
    #     start_frame_time = time.time()
    #     ok, frame = video_reader.read()
    #     if not ok:
    #         break
    #     if not frame.size:
    #         continue
    #     end_frame_time = time.time()
    #     times['read_frame'] = times['read_frame'] + (end_frame_time - start_frame_time)

    #     start_resize_time = time.time()
    #     resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    #     end_resize_time = time.time()
    #     times['resize_frame'] = times['resize_frame'] + (end_resize_time - start_resize_time)

    #     start_normalize_time = time.time()
    #     normalized_frame = resized_frame / 255.0
    #     end_normalize_time = time.time()
    #     times['normalize_frame'] = times['normalize_frame'] + (end_normalize_time - start_normalize_time)

    #     frames_queue.append(normalized_frame)
    #     if len(frames_queue) == SEQUENCE_LENGTH:
            
    #         frame_list = list(frames_queue)
    #         input_list = [frame_list]
    #         input_tensor = torch.tensor(input_list).to(device).float()
    #         model = model.to(device)
    #         start_model_time = time.time()
    
    #         predicted_labels_probabilities = model(input_tensor)
    #         end_model_time = time.time()
    #         probability_tensor = F.softmax(predicted_labels_probabilities, dim=1)
    #         probability_array = probability_tensor.detach().cpu().numpy()
            
    #         times['model_inference'] = times['model_inference'] + (end_model_time - start_model_time)

    #         probability = probability_array[0]
    #         predicted_label = np.argmax(probability)
    #         predicted_confidence = probability[predicted_label]
    #         if predicted_confidence >= confidence_threshold:
    #             predicted_class_name = CLASSES_LIST[predicted_label]
    #             cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    #     start_frame_time=time.time()
    #     video_writer.write(frame)
    #     end_write_time = time.time()
    #     times['write_frame'] =times['write_frame'] + (end_write_time - start_frame_time)

    # video_reader.release()
    # video_writer.release()

    # # 计算总时间
    # total_time = times['write_frame']+times['model_inference']+times['normalize_frame']+times['resize_frame']+times['read_frame']

    # # 打印时间占比
    # for key, time_spent in times.items():
    #     print(f"{key.replace('_', ' ').title()} Time: {time_spent:.4f} seconds,占比: {(time_spent / total_time) * 100:.2f}%")

    # print(f"Total Processing Time: {total_time:.4f} seconds")

import time
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
    output_video_name = args.output_video_name
    output_video_file_path = os.path.join(output_dir, output_video_name)
    predict_on_video(input_video_file_path, output_video_file_path, args.sequence_length, model,device, args.classes_list,args.image_height,args.image_width)
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

    args = parser.parse_args()
    main(args)
