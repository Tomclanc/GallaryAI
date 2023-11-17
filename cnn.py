import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


# 检测是否存在GPU，并据此设置device变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 禁用解压缩炸弹检测（即解除像素数量限制）
Image.MAX_IMAGE_PIXELS = None

# 图像预处理（包括数据增强）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []

        # 定义各个文件夹的绝对路径
        people_dir_1 = r'D:\OneDrive\图片\手机图片\1'
        people_dir_2 = r'D:\OneDrive\图片\手机图片\2'
        landscape_dir = r'D:\OneDrive\图片\手机图片\2023'

        # 临时列表用于构建数据集
        temp_samples = []

        # 为人物图片分配标签 0
        for people_dir in [people_dir_1, people_dir_2]:
            temp_samples.extend([(os.path.join(people_dir, f), 0) for f in os.listdir(people_dir) if f.endswith(('.jpg', '.png'))])

        # 为风景图片分配标签 1
        temp_samples.extend([(os.path.join(landscape_dir, f), 1) for f in os.listdir(landscape_dir) if f.endswith(('.jpg', '.png'))])

        # 过滤掉无法打开的图像
        for path, label in temp_samples:
            try:
                with Image.open(path):
                    self.samples.append((path, label))
            except (IOError, SyntaxError):
                print('Bad file skipped:', path)


      
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except (IOError, SyntaxError) as e:
            print('Bad file:', path)
            return None

        return image, label

# 使用预训练的 ResNet 模型
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet(x)


# 训练模型
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置TensorBoard日志目录为当前脚本所在目录
    writer = SummaryWriter(script_dir)

    best_acc = 0.0  # 初始化最佳准确率

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            # 将数据移至GPU
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        # 使用TensorBoard记录损失和准确率
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

       # 如果当前周期的准确率是最好的，保存模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 保存模型到当前脚本所在目录
            model_save_path = os.path.join(script_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
    writer.close()  # 关闭TensorBoard记录器    

# 加载数据集
dataset = CustomDataset(transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型并移至GPU
model = ResNetModel().to(device)  # 修改这里

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 开始训练
train_model(model, dataloader, criterion, optimizer)