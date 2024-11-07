import argparse
import os
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras.models import load_model
from moviepy.editor import VideoFileClip
from collections import deque
from utils import extract_keyframes, Timer

def predict_on_video(video_file_path, output_file_path, model, sequence_length, image_height, image_width, classes_list, confidence_threshold=0.8):
    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                  video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frames_queue = deque(maxlen=sequence_length)
    index = 0
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break  # Exit the loop when there are no more frames

        # Check if the frame is empty before resizing
        if not frame.size:
            continue

        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == sequence_length:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)

            # 增加置信度判断
            predicted_confidence = predicted_labels_probabilities[predicted_label]

            if predicted_confidence >= confidence_threshold:
                predicted_class_name = classes_list[predicted_label]
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join('./output2', f"{predicted_class_name}_{index}.jpg"), frame)
                index += 1

            video_writer.write(frame)
    print('down')
    video_reader.release()
    video_writer.release()


def predict_on_video_v2(
    video_file_path,
    output_file_path,
    model,
    sequence_length,
    image_height,
    image_width,
    classes_list,
    confidence_threshold=0.8,
    method="LOCAL_MAXIMA",
    num_top_frames=50,
    threshold=0.6,
    len_window=50
):
    key_frames = extract_keyframes(
        video_file_path,
        output_file_path,
        method=method,
        save=False,
        num_top_frames=num_top_frames,
        threshold=threshold,
        len_window=len_window
    )
    frames_queue = deque(maxlen=sequence_length)

    for index, frame in enumerate(key_frames):
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)
        if len(frames_queue) == sequence_length:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)

            # 增加置信度判断
            predicted_confidence = predicted_labels_probabilities[predicted_label]

            if predicted_confidence >= confidence_threshold:
                predicted_class_name = classes_list[predicted_label]
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(output_file_path, f"{predicted_class_name}_{index}.jpg"), frame)
                
        if index == len(key_frames) - 1 and len(frames_queue) < sequence_length:
            # 当提取的关键帧数量小于sequence_length, 在队尾填充空白帧
            blank_frame = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            while len(frames_queue) < sequence_length:
                frames_queue.append(blank_frame)
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)

            # 增加置信度判断
            predicted_confidence = predicted_labels_probabilities[predicted_label]

            if predicted_confidence >= confidence_threshold:
                predicted_class_name = classes_list[predicted_label]
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(output_file_path, f"{predicted_class_name}_{index}.jpg"), frame)
        
    print('down')


def main(args):
    # Load the LRCN model from a local file
    print('model loading')
    model = load_model(args.model_path)

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    timer = Timer()
    timer.start
    if args.extract_keyframe:
        predict_on_video_v2(args.input_video_file_path, output_dir, model, args.sequence_length, args.image_height, args.image_width, args.classes_list, confidence_threshold=0.8, method=args.method, num_top_frames=args.num_top_frames, threshold=args.threshold, len_window=args.len_window)
        
    else: 
        # Predict on video
        output_video_file_path = os.path.join(output_dir, args.output_video_name)
        predict_on_video(args.input_video_file_path, output_video_file_path, model, args.sequence_length, args.image_height, args.image_width, args.classes_list)

        # Process and display the video
        
        processed_video = VideoFileClip(output_video_file_path, audio=False, target_resolution=(args.target_resolution, None))
        processed_video.write_videofile(output_video_file_path, codec='libx264', audio=False)
        # processed_video.ipython_display(maxduration=1800)
        
    cost = timer.stop()
    print(f'Processing time: {cost} seconds')

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
    parser.add_argument('--extract_keyframe', action='store_true', help='Whether to extract keyframes.')
    parser.add_argument("--method",choices=["TOP_ORDER", "Threshold", "LOCAL_MAXIMA"],default="LOCAL_MAXIMA",help="The method to extract key frames: 'TOP_ORDER','Threshold', or 'LOCAL_MAXIMA'")
    parser.add_argument("--num_top_frames", type=int, default=50, help="The number of top frames to extract if using the 'TOP_ORDER' method.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Used in the 'Threshold' method to define the minimum relative change between consecutive frames to consider a frame as a keyframe.")
    parser.add_argument("--len_window", type=int, default=50, help="The window size for smoothing frame differences in the 'LOCAL_MAXIMA' method")

    args = parser.parse_args()
    main(args)
