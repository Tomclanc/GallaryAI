import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import os
import shutil

# 增加Pillow的像素限制以避免解压缩炸弹错误
Image.MAX_IMAGE_PIXELS = None

# 检测是否存在GPU，并据此设置device变量
print("检测 GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义模型
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet(x)

# # 初始化模型并尝试载入权重
model = ResNetModel().to(device)

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best_model.pth')

# 检查模型文件是否存在
if os.path.exists(model_path):
    print("找到模型文件，正在加载...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("模型加载完毕。")
else:
    print("模型文件不存在。")

# 图像预处理
print("设置图像预处理...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定义类别
classes = ['Person', 'Landscape']  # 根据您的类别标签修改

# 文件夹路径，根据个人需要修改相册文件夹的位置，使用绝对路径
folder_path = 'D:\\OneDrive\\iPad照片\\2023'

# 创建目标文件夹，若有需要修改绝对路径
os.makedirs('D:\\OneDrive\\iPad照片\\2023\\Person', exist_ok=True)
os.makedirs('D:\\OneDrive\\iPad照片\\2023\\Landscape', exist_ok=True)

#设置阈值
threshold = 0.5  # 设置置信度阈值

# 确保目标文件夹存在，需要修改为所需的绝对路径
target_folder_person = 'D:\\OneDrive\\iPad照片\\2023\\Person'
target_folder_landscape = 'D:\\OneDrive\\iPad照片\\2023\\Landscape'
os.makedirs(target_folder_person, exist_ok=True)
os.makedirs(target_folder_landscape, exist_ok=True)

# 遍历文件夹中的所有图片
print("开始处理图片...")
try:
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png')):
            file_count += 1
            image_path = os.path.join(folder_path, filename)
            print(f"处理文件: {image_path}")

            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                image = image.to(device)
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, predicted = torch.max(probabilities, 1)
                
                # 修改置信度判断逻辑
                if max_prob.item() < threshold:
                    print(f'Image: {filename}, Prediction: Uncertain with confidence {max_prob.item()}')
                    continue  # 跳过当前循环，不进行分类

                predicted_class = classes[predicted[0]]
                print(f'Image: {filename}, Predicted: {predicted_class} with confidence {max_prob.item()}')

                # 根据预测结果移动文件
                target_folder = os.path.join('D:\\OneDrive\\iPad照片\\2023', predicted_class)
                target_path = os.path.join(target_folder, filename)
                print(f'Moving {image_path} to {target_path}')
                shutil.move(image_path, target_path)
                print(f'Moved {filename} to {target_folder}')

    if file_count == 0:
        print("没有找到图片文件。")
except Exception as e:
    print(f"发生错误: {e}")
