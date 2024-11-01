import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

class LSTMModel:
    def __init__(self, dataset_dir, classes_list, image_height, image_width, sequence_length):
        self.dataset_dir = dataset_dir
        self.classes_list = classes_list
        self.image_height = image_height
        self.image_width = image_width
        self.sequence_length = sequence_length
        self.seed_constant = 27

    def frames_extraction(self, video_path):
        frames_list = []
        video_reader = cv2.VideoCapture(video_path)
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count / self.sequence_length), 1)
        for frame_counter in range(self.sequence_length):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, (self.image_height, self.image_width))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)
        video_reader.release()
        return frames_list

    def create_dataset(self):
        features = []
        labels = []
        video_files_paths = []
        for class_index, class_name in enumerate(self.classes_list):
            print(f'Extracting Data of CLass: {class_name}')
            files_list = os.listdir(os.path.join(self.dataset_dir, class_name))
            for file_name in files_list:
                video_file_path = os.path.join(self.dataset_dir, class_name, file_name)
                frames = self.frames_extraction(video_file_path)
                if len(frames) == self.sequence_length:
                    features.append(frames)
                    labels.append(class_index)
                    video_files_paths.append(video_file_path)
        features = np.asarray(features)
        labels = np.array(labels)
        return features, labels, video_files_paths

    def create_LRCN_model(self):
        model = Sequential()
        model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'),
                                  input_shape=(self.sequence_length, self.image_height, self.image_width, 3)))
        model.add(TimeDistributed(MaxPooling2D((4, 4))))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((4, 4))))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(32))
        model.add(Dense(len(self.classes_list), activation='softmax'))
        model.summary()
        return model

    def train(self, features, labels):
        np.random.seed(self.seed_constant)
        tf.random.set_seed(self.seed_constant)
        features_train, features_test, labels_train, labels_test = train_test_split(features, to_categorical(labels), test_size=0.25, shuffle=True, random_state=self.seed_constant)
        LRCN_model = self.create_LRCN_model()
        early_stopping_callback = EarlyStopping(monitor='accuracy', patience=10, mode='max', restore_best_weights=True)
        LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
        LRCN_model_training_history = LRCN_model.fit(x=features_train, y=labels_train, epochs=100, batch_size=4, shuffle=True, validation_split=0.2, callbacks=[early_stopping_callback])
        return LRCN_model_training_history, features_test, labels_test

    def evaluate(self, model, features_test, labels_test):
        model_evaluation_history = model.evaluate(features_test, labels_test)
        return model_evaluation_history

    def plot_metric(self, model_training_history, metric_name1, metric_name2, plot_name):
        metric_value1 = model_training_history.history[metric_name1]
        metric_value2 = model_training_history.history[metric_name2]
        epochs = range(len(metric_value1))
        plt.plot(epochs, metric_value1, 'blue', label=metric_name1)
        plt.plot(epochs, metric_value2, 'red', label=metric_name2)
        plt.title(str(plot_name))
        plt.legend()

    def save_model(self, model, model_file_name):
        model.save(model_file_name)