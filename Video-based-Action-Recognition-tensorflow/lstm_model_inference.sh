python lstm_model_inference.py \
--model_path "./LSTM_model_2024_11_01_14_04_18.h5" \
--input_video_file_path "./test_videos/20241025_105625.mp4" \
--output_dir "./Output" \
--output_video_name "output.mp4" \
--sequence_length 20 \
--classes_list openfile1 openmail1 \
--extract_keyframe \