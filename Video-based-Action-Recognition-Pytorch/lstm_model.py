import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import cv2
import torch.nn.functional as F
import torch.nn as nn
import time
class LSTMModel:
    def __init__(self, dataset_dir, classes_list, image_height, image_width, sequence_length):
        self.dataset_dir = dataset_dir
        self.classes_list = classes_list
        self.image_height = image_height
        self.image_width = image_width
        self.sequence_length = sequence_length
        self.seed_constant = 27
        print('test')
        self.num_classes=len(classes_list)

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

    def create_dataset(self,classes_list,dataset_dir):
        features = []
        labels = []
        video_files_paths = []
        for class_index, class_name in enumerate(classes_list):
            print(f'Extracting Data of CLass: {class_name}')
            files_list = os.listdir(os.path.join(dataset_dir, class_name))
            for file_name in files_list:
                video_file_path = os.path.join(dataset_dir, class_name, file_name)
                frames = self.frames_extraction(video_file_path)
                if len(frames) == self.sequence_length:
                    features.append(frames)
                    labels.append(class_index)
                    video_files_paths.append(video_file_path)
        features = np.asarray(features)
        labels = np.array(labels)
        return features, labels, video_files_paths

    def create_LRCN_model(self):
        class LRCN(nn.Module):
            def __init__(self, image_height=64, image_width=64, num_classes=2):
                super(LRCN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
                self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
                self.pool = nn.MaxPool2d((4, 4))
                self.dropout = nn.Dropout(0.25)
                self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
                self.fc = nn.Linear(32, num_classes)

            def forward(self, x):
                # x 的形状是 (batch_size, sequence_length, channels, height, width)
                # 需要将 x 重新排列为 (batch_size * sequence_length, channels, height, width)
                
                batch_size, sequence_length,  H, W,C = x.shape
                x = x.view(-1, C, H, W)  # 展平序列维度
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = F.relu(self.conv2(x))
                x = self.pool(x)
                x = F.relu(self.conv3(x))
                x = self.pool(x)
                x = F.relu(self.conv4(x))
                
                # 将 x 重新排列回 (batch_size, sequence_length, -1)
                x = x.view(batch_size, sequence_length, -1)  # 这里的-1代表自动计算特征维度
                
                # LSTM 期望的输入形状是 (batch_size, sequence_length, features)
                x, _ = self.lstm(x)
                x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
                return x
        return LRCN(image_height=self.image_height, image_width=self.image_width, num_classes=self.num_classes)

    def train(self,model, train_loader, test_loader,device,epochs=100,early_stopping_patience=10):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model=model.to(torch.float32)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        start_time = time.time()
        best_accuracy=0
        for epoch in range(epochs):
            # model.train()
            running_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                features = features.to(device).to(torch.float32)  # 确保数据为float32
                #print(features.shape)
                labels = labels.to(device).to(torch.float32)  # 确保标签也为float32 
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

            # 评估模型
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    features = features.to(device).to(torch.float32)  # 确保数据为float32
                    labels = labels.to(device).to(torch.float32)  # 确保标签也为float32 
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    # print(predicted)
                    total += labels.size(0)
                    correct += (predicted == labels.long()).sum().item()
            accuracy = 100 * correct / total
            print(f'Accuracy: {accuracy:.2f}%')

            # 早停逻辑
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping at epoch"+epoch)
                    break

        # 训练后的时间
        end_time = time.time()

        # 计算总训练时间
        total_training_time = end_time - start_time
        print(f"Total training time: {total_training_time:.2f} seconds")
        return model