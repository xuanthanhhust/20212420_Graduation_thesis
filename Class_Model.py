import torch.nn.functional as F
from torch.backends import cudnn
import torch.nn as nn
import torch
import torchvision.models as models

#Cấu hình PyTorch
cudnn.benchmark = False
cudnn.deterministic = True #Đảm bảo rằng mô hình luôn tạo ra kết quả giống nhau khi huấn luyện lại.

torch.manual_seed(0)

# Xây dựng mô hình ResNet
# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.num_classes = 4 

#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         # Tính toán kích thước đầu ra của self.conv
#         dummy_input = torch.zeros(1, 1, 100, 100)  # Giả sử đầu vào là (1, 1, 100, 100)
#         conv_output_size = self._get_conv_output_size(dummy_input)

#         self.fc = nn.Sequential(
#             nn.Linear(conv_output_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(128, self.num_classes)
#         )

# #     def _get_conv_output_size(self, x):
# #         x = self.conv(x)
# #         return int(torch.prod(torch.tensor(x.shape[1:])))

# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = x.view(x.size(0), -1)  # Flatten
# #         x = self.fc(x)
# #         return x

# class MyModule(nn.Module):
#     def __init__(self, num_classes=4):
#         super(MyModule, self).__init__()
#         # Dùng ResNet18, thay đổi input conv để nhận 1 channel
#         self.model = models.resnet18(pretrained=False)
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # Thay đổi lớp fully connected cuối để phù hợp số lớp
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)
    

# class MyModule(nn.Module): 
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.num_classes = 4

#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # Đầu vào 1 kênh
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )


#         # Tính flatten_size động (kể cả sau khi prune)
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, 1, 100, 100)
#             conv_output_size = self._get_conv_output_size(dummy_input)

#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(conv_output_size , 4096), # +4 là kích thước của RR interval
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, self.num_classes)
#         )


class MyModule(nn.Module):
    def __init__(self):

        super(MyModule, self).__init__()
        self.num_classes = 4
        # Tải EfficientNet-B0, không pretrained vì input là 1 channel
        self.model = models.efficientnet_b0(weights=None)
        # Sửa lớp đầu vào cho ảnh 1 kênh
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)


    def forward(self, x):
        return self.model(x)


# class MyModule(nn.Module):
#     def __init__(self, num_classes=4):
#         super(MyModule, self).__init__()
#         self.model = models.resnet18(weights=None)
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)


#     # ALexNet dowload 
# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.num_classes = 4
#         self.model = models.alexnet(weights=None)
#         self.model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # Đầu vào 1 kênh
#         with torch.no_grad():
#             dummy = torch.zeros(1, 1, 100, 100)
#             x = self.model.features(dummy)
#             x = self.model.avgpool(x)
#             flatten_size = x.view(1, -1).size(1)
#         # Sửa toàn bộ classifier để phù hợp flatten_size
#         self.model.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(flatten_size, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, self.num_classes)
#         )
#     def forward(self, x):
#         x = self.model.features(x)
#         x = self.model.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.model.classifier(x)
#         return x

# MobileNetV2 
# class MyModule(nn.Module):
#     def __init__(self, num_classes=4):
#         super(MyModule, self).__init__()
#         self.num_classes = num_classes
        
#         # Load MobileNetV2
#         self.model = models.mobilenet_v2(weights=None)
        
#         # Thay đổi layer đầu vào để nhận ảnh 1 kênh
#         self.model.features[0][0] = nn.Conv2d(
#             in_channels=1, out_channels=32, kernel_size=3,
#             stride=2, padding=1, bias=False
#         )
        
#         # Tính kích thước đầu ra sau phần features
#         with torch.no_grad():
#             dummy = torch.zeros(1, 1, 100, 100)  # ảnh giả đầu vào
#             x = self.model.features(dummy)
#             flatten_size = x.shape[1] * x.shape[2] * x.shape[3]

#         # Thay classifier (head)
#         self.model.classifier = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(flatten_size, 1280),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.2),
#             nn.Linear(1280, self.num_classes)
#         )

#     def forward(self, x):
#         x = self.model.features(x)
#         x = x.view(x.size(0), -1)  # flatten
#         x = self.model.classifier(x)
#         return x
    

#ResNet 18 
# class MyModule(nn.Module):
#     def __init__(self, num_classes=4):
#         super(MyModule, self).__init__()
        
#         # Load pretrained ResNet18, bỏ lớp fully connected cuối
#         self.model = models.resnet18(weights=None)
        
#         # Thay đổi conv đầu tiên để nhận ảnh 1 kênh (grayscale)
#         self.model.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=64, kernel_size=7,
#             stride=2, padding=3, bias=False
#         )
        
#         # Thay đổi fully connected layer cuối để phù hợp số lớp đầu ra
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)