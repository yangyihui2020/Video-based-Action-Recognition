import argparse
import os
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.models import load_model
from moviepy.editor import VideoFileClip
from collections import deque

def predict_on_video(video_file_path, output_file_path, model, sequence_length, image_height, image_width, classes_list, confidence_threshold=0.8):
    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                  video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frames_queue = deque(maxlen=sequence_length)

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

            video_writer.write(frame)
    print('down')
    video_reader.release()
    video_writer.release()

def main(args):
    # Load the LRCN model from a local file
    print('model loading')
    model = load_model(args.model_path)

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Predict on video
    output_video_file_path = os.path.join(output_dir, args.output_video_name)
    predict_on_video(args.input_video_file_path, output_video_file_path, model, args.sequence_length, args.image_height, args.image_width, args.classes_list)

    # Process and display the video
    
    processed_video = VideoFileClip(output_video_file_path, audio=False, target_resolution=(args.target_resolution, None))
    processed_video.write_videofile(output_video_file_path, codec='libx264', audio=False)
    # processed_video.ipython_display(maxduration=1800)

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