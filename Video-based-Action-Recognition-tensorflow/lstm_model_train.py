import argparse
# 不打印tensorflow官方的提示信息
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from lstm_model import LSTMModel
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import numpy as np
import random
import tensorflow as tf


def main(args):
    # 设置随机种子以确保结果可复现
    seed_constant = args.seed
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    tf.random.set_seed(seed_constant)

    # 初始化 LSTMModel 类
    lstm_model = LSTMModel(args.dataset_dir, args.classes_list, args.image_height, args.image_width, args.sequence_length)

    # 创建数据集
    print('Begin creating dataset')
    features, labels, video_files_paths = lstm_model.create_dataset(args.extract_keyframe)
    print('Dataset created successfully')

    # 将标签转换为 one-hot 编码
    one_hot_encoded_labels = to_categorical(labels)

    # 划分训练集和测试集
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, one_hot_encoded_labels, test_size=0.25, shuffle=True, random_state=seed_constant
    )

    # 创建 LRCN 模型
    LRCN_model = lstm_model.create_LRCN_model()
    print("Model created successfully!")

    # 训练前的准备
    start_time = time.time()

    # 设置早停回调函数
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, mode='max', restore_best_weights=True)

    # 编译模型
    LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

    # 开始训练
    LRCN_model_training_history = LRCN_model.fit(
        x=features_train, y=labels_train, epochs=args.epochs, batch_size=args.batch_size,
        shuffle=True, validation_split=0.2, callbacks=[early_stopping_callback]
    )

    # 训练后的时间
    end_time = time.time()

    # 计算总训练时间
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # 评估模型
    model_evaluation_history = LRCN_model.evaluate(features_test, labels_test)

    # 保存模型
    current_date_time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_file_name = f'LSTM_model_{current_date_time_string}.h5'
    lstm_model.save_model(LRCN_model, model_file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a LSTM model for video classification.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory path to the dataset.')
    parser.add_argument('--classes_list', type=str, nargs='+', required=True, help='List of class names.')
    parser.add_argument('--image_height', type=int, default=64, help='Height of the images.')
    parser.add_argument('--image_width', type=int, default=64, help='Width of the images.')
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of the sequence.')
    parser.add_argument('--seed', type=int, default=27, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--extract_keyframe', action='store_true', help='Whether to extract keyframes.')

    args = parser.parse_args()
    main(args)