import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, random_split
from lstm_model import LSTMModel
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import issparse

def main(args):
    # 设置随机种子以确保结果可复现
    seed_constant = 27
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    torch.manual_seed(seed_constant)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_constant)
    print('program begin')
    # 初始化 LSTMModel 类

    # 创建数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    lstm_model=LSTMModel(dataset_dir=args.dataset_dir,
                         classes_list=args.classes_list,
                         image_height=args.image_height,
                         image_width=args.image_width,
                         sequence_length=args.sequence_length)

    print('Begin creating dataset')
    features, labels, _ = lstm_model.create_dataset(classes_list=args.classes_list,dataset_dir=args.dataset_dir)
    print('Dataset created successfully')
    print(features.shape)
    print(labels.shape)
    print(labels)

    features_tensor = torch.from_numpy(features)
    labels_tensor = torch.from_numpy(labels)
    dataset = TensorDataset(features_tensor, labels_tensor)

    # 分割数据集    
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 创建 LRCN 模型
    model = lstm_model.create_LRCN_model()
    print("Model created successfully!")

    # 开始训练
    torch.backends.cudnn.enabled = False
    model=lstm_model.train(model,train_loader,test_loader,device,100)

    # 保存
    model_file_name = f'LSTM_model.pth'
    torch.save(model.state_dict(), model_file_name)

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

    args = parser.parse_args()
    main(args)

