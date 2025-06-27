import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from tqdm import tqdm

import time

from Class_Dataset import MyDataset
from Class_Model import MyModule

import matplotlib.pyplot as plt
import seaborn as sns
###########################    CONFIGURE   #############################

DATA_ROOT_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/data")
MODEL_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/model_save/ecg_model.pth")
COMPACT_MODEL_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/model_save/ecg_model_compact.pth")
QUANTIZED_MODEL_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/model_save/ecg_model_quantized_1.pth")
BATCH_SIZE = 128

# ====================== Inference & Evaluation ======================

def evaluate():
    torch.serialization.add_safe_globals([MyModule])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    #model = MyModule()
    model = torch.load(MODEL_PATH, weights_only=False)
    # model = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    #model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load test data
    test_dataset = MyDataset(root_dir=DATA_ROOT_PATH/'test_images', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    start_time = time.time() #Bắt đầu tính thời gian

    #Là chạy mô hình trên tập test, lấy dự đoán (preds) và so sánh với nhãn thật (labels)
    with torch.no_grad(): #Tắt chế độ tính gradient để tiết kiệm bộ nhớ và tăng tốc, ko cần học 
        for inputs, labels in tqdm(test_loader, desc="🔍 Testing"): #inputs: batch ảnh scalogram, labels: nhãn tương ứng (class thực tế)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) #Dữ liệu đi qua mô hình (Nếu mô hình có lớp softmax/logits ở cuối, outputs sẽ là [batch_size, num_classes])
            _, preds = torch.max(outputs, 1) #Lấy chỉ số class có giá trị lớn nhất ở mỗi hàng → chính là class mà mô hình dự đoán
                #preds: vector dự đoán class (vd: [0, 1, 2, ...])

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            #preds và labels được chuyển về CPU và numpy để: Dễ dùng với sklearn (dùng tính accuracy, confusion matrix, F1-score).
    end_time = time.time() #Kết thúc tính thời gian
    num_images = len(all_labels) #Số lượng ảnh trong tập test
    total_time = end_time - start_time #Tính thời gian chạy
    print(f"\nĐã predict {num_images} ảnh trong {total_time:.2f} giây")
    print(f"Tốc độ: {(num_images/total_time):.2f} ảnh/s")   

    # Evaluation 
    print("✅ Evaluation Result:") #In ra tiêu đề
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4)) #digits=4: làm tròn đến 4 chữ số thập phân

    #Tính Macro F1-score cho từng lớp, rồi lấy trung bình cộng
    print(f"F1 Score (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}") 

    #Weighted F1: Tính F1-score từng lớp, nhưng cân theo số lượng mẫu ở mỗi lớp
    #Phù hợp khi các lớp mất cân bằng (unbalanced)
    print(f"F1 Score (weighted): {f1_score(all_labels, all_preds, average='weighted'):.4f}")

    # Vẽ confusion matrix dạng phần trăm với heatmap
    cm_percent = cm.astype('float')
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', cbar=True, square=True)
    plt.title('Confusion Matrix (%)')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()
        # Lưu kết quả ra file txt
    with open("test_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Đã predict {num_images} ảnh trong {total_time:.2f} giây\n")
        f.write(f"Tốc độ: {(num_images/total_time):.2f} ảnh/s\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(all_labels, all_preds, digits=4))
        f.write(f"\nF1 Score (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}\n")
        f.write(f"F1 Score (weighted): {f1_score(all_labels, all_preds, average='weighted'):.4f}\n")

    print("✅ Đã lưu kết quả vào test_result.txt")
if __name__ == '__main__':
    evaluate()
