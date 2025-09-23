import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {
            'straight': 0,
            'left': 1,
            'right': 2,
            'stop': 3
        }

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]  # CSV에서 이미지 경로 가져오기
        image = Image.open(img_name)
        label_name = self.labels_df.iloc[idx, 1]
        label = self.label_map.get(label_name, -1)  # 라벨 매핑에 없는 경우 -1 반환

        if self.transform:
            image = self.transform(image)

        return image, label


# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((70, 320)),
    transforms.ToTensor()  # 흑백 이미지 그대로 유지 (1 채널)
])

# 데이터셋 및 데이터로더 준비
train_dataset = CustomDataset(csv_file='4train.csv', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 모델 정의 (더 깊은 Conv 및 Linear 레이어 추가)
class NetworkNvidia(nn.Module):
    def __init__(self):
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),  # 첫 번째 Conv 레이어
            nn.ELU(),
            nn.BatchNorm2d(16),             # Batch Normalization 추가
            nn.Conv2d(16, 32, 5, stride=2), # 두 번째 Conv 레이어
            nn.ELU(),
            nn.BatchNorm2d(32),             # Batch Normalization 추가
            nn.Conv2d(32, 64, 3),           # 세 번째 Conv 레이어
            nn.ELU(),
            nn.BatchNorm2d(64),             # Batch Normalization 추가
            nn.Conv2d(64, 128, 3),          # 네 번째 Conv 레이어
            nn.ELU(),
            nn.BatchNorm2d(128),            # Batch Normalization 추가
            nn.Conv2d(128, 256, 3),         # 다섯 번째 Conv 레이어 추가
            nn.ELU(),
            nn.BatchNorm2d(256),            # Batch Normalization 추가
            nn.Dropout(0.5)                 # 드롭아웃 적용 (0.5로 설정)
        )

        # Conv 레이어 이후 Linear 레이어 추가
        self.fc_input_size = self._get_conv_output((1, 70, 320))  # 입력 이미지 크기 (1, 70, 320)
        self.linear_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 500),  # Conv 레이어의 출력을 Flatten한 후 고정된 입력 크기
            nn.ELU(),
            nn.BatchNorm1d(500),           # Batch Normalization 추가
            nn.Linear(500, 100),           # 중간층 추가
            nn.ELU(),
            nn.BatchNorm1d(100),           # Batch Normalization 추가
            nn.Linear(100, 50),            # 중간층 추가
            nn.ELU(),
            nn.BatchNorm1d(50),            # Batch Normalization 추가
            nn.Linear(50, 10),             # 중간층
            nn.ELU(),
            nn.Linear(10, 4)               # 4개 클래스 출력
        )

    def _get_conv_output(self, shape):
        """Conv 레이어의 출력 크기를 계산하는 도우미 함수"""
        input = torch.rand(1, *shape)  # 임의의 입력을 만들어 Conv 레이어를 통과시킴
        output = self.conv_layers(input)
        return int(torch.flatten(output, 1).size(1))  # Flatten한 후의 출력 크기 반환

    def forward(self, input):
        input = input.view(input.size(0), 1, 70, 320)  # 흑백 이미지 (1 채널)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)       # Flatten
        output = self.linear_layers(output)            # Linear 레이어로 예측
        return output


# 장치 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 모델, 손실 함수, 최적화기 정의
model = NetworkNvidia().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 모델 학습
num_epochs = 10  # 학습 에포크 수 설정
losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = correct / total

    losses.append(epoch_loss)
    accuracies.append(accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

# 모델 저장
torch.save(model.state_dict(), 'model_4class_upgrade4.pth')

print("모델 학습 완료 및 저장.")

# 손실 및 정확도 그래프 그리기
plt.figure(figsize=(12, 5))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy', color='orange')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
