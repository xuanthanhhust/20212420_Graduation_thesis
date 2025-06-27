import torch
import torch.nn as nn
import torchvision
import torch_pruning as tp
from torchvision import datasets, transforms
from tqdm import tqdm
from Class_Model import MyModule
from Class_Dataset import MyDataset
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from torch.backends import cudnn
from thop import profile  # Để tính FLOPs
from torchsummary import summary  # Để xem kích thước mô hình
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA_ROOT_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/data")
MODEL_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/model_save/ecg_model.pth")
COMPACT_MODEL_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/model_save/ecg_model_compact.pth")
BATCH_SIZE = 128


# 2. Hàm tính FLOPs và Model Size
def evaluate_model(model, input_shape=(1, 100, 100)):
    # Tính FLOPs
    input_tensor = torch.randn(1, *input_shape).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    
    # Tính Model Size (MB)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = (param_size + buffer_size) / (1024 ** 2)  # Convert sang MB
    
    return flops, params, model_size

# 1. Load dữ liệu CIFAR-10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình và đưa lên device
# model = MyModule().to(device)
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device)
model.eval()
    # Tạo DataLoader cho dữ liệu huấn luyện
transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
train_dataset = MyDataset(root_dir=DATA_ROOT_PATH/'train_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = MyDataset(root_dir=DATA_ROOT_PATH/'test_images', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Định nghĩa hàm eval để tính Fisher Information
def eval_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return round((correct / total), 4)

# 4. Tính toán Fisher Information và Prune
def pruning(model, train_loader, prune_ratio=0.5):
    # Khởi tạo FisherPruner
    example_inputs = torch.randn(1, 1, 100, 100).to(device)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=tp.importance.MagnitudeImportance(),
        ch_sparsity= prune_ratio,
        # ignored_layers=[model.fc[-1]] # Dành cho model mà có lớp cuối là fc 
        ignored_layers=[model.model.classifier[1]] #EfficientNet.b0
        # ignored_layers= [model.model.classifier[6]] #AlexNet
        # ignored_layers=[model.model.fc]  # ResNet18


    )

    # Tính Fisher Information bằng cách chạy 1 epoch
    # for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     pruner.accumulate_gradients(inputs, targets)  # Cập nhật Fisher scores

    # Thực hiện pruning
    pruner.step()  # Prune 50% weights

    return model

def save_model_structure(model, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for name, module in model.named_modules():
            f.write(f"{name}: {module}\n")

def print_main_layers(model, title):
    print(f"\n===== {title} =====")
    for name, module in model.named_children():
        print(f"{name}: {module}")

# 5. Thực hiện Pruning và kiểm tra độ chính xác

print("=== Trước khi prune ===")
flops_before, params_before, size_before = evaluate_model(model)
print(f"FLOPs: {flops_before / 1e6:.2f}M | Params: {params_before / 1e3:.2f}K | Size: {size_before:.2f}MB")
# Trước khi prune
save_model_structure(model, "model_structure_before.txt")


pruned_model = pruning(model, train_loader)

# Sau khi prune
save_model_structure(pruned_model, "model_structure_after.txt")



flops_after, params_after, size_after = evaluate_model(pruned_model)
print(f"FLOPs: {flops_after / 1e6:.2f}M | Params: {params_after / 1e3:.2f}K | Size: {size_after:.2f}MB")
# print("Accuracy sau khi prune:", eval_model(pruned_model, train_loader, device))
# 6. So sánh hiệu quả
print("\n=== So sánh ===")
print(f"FLOPs giảm: {(flops_before - flops_after) / flops_before * 100:.2f}%")
print(f"Params giảm: {(params_before - params_after) / params_before * 100:.2f}%")
print(f"Model Size giảm: {(size_before - size_after) / size_before * 100:.2f}%") 
# Lưu thông số FLOPs, Params, Model Size trước và sau prune vào file
with open("prune_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== Pruning Summary ===\n")
    f.write(f"FLOPs trước: {flops_before/1e6:.2f}M | sau: {flops_after/1e6:.2f}M | giảm: {(flops_before-flops_after)/flops_before*100:.2f}%\n")
    f.write(f"Params trước: {params_before/1e3:.2f}K | sau: {params_after/1e3:.2f}K | giảm: {(params_before-params_after)/params_before*100:.2f}%\n")
    f.write(f"Model Size trước: {size_before:.2f}MB | sau: {size_after:.2f}MB | giảm: {(size_before-size_after)/size_before*100:.2f}%\n")

print("✅ Đã lưu thông số prune vào prune_summary.txt")

# 6. (Optional) Fine-tuning sau pruning
optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):  # Fine-tune 5 epochs
    pruned_model.train()
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = pruned_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Accuracy:", eval_model(pruned_model, train_loader, device))

torch.save(pruned_model, COMPACT_MODEL_PATH)
print(f"\nPruned model saved to: {COMPACT_MODEL_PATH}")
# 7. Đánh giá model đã prune trên tập test
def test_model(model, test_loader, device):
    print("Đang test mô hình đã prune...")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # total += targets.size(0)
            # correct += (predicted == targets).sum().item()
    # acc = correct / total
    # print(f"Test accuracy: {acc*100:.2f}%")
    # return acc
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
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', cbar=True, square=True)
    plt.title('Confusion Matrix (%)')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()
    

# Giả sử bạn đã có test_loader
test_model(pruned_model, test_loader, device)

