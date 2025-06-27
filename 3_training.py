# HELLO THANHVERDA
# ê thành ơi t ctrl c nha
#ai bày m chạy 200 epooch đó 



import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch.backends import cudnn
import torch.nn as nn
import torch
import torch.optim as optim 
from pathlib import Path
from tqdm import tqdm

from Class_Dataset import MyDataset
from Class_Model import MyModule

from collections import Counter

DATA_ROOT_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/data")
OUTPUT_MODEL_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/model_save/ecg_model.pth")
batch_size = 64
num_epochs = 10

# Áp dụng các biến đổi cho ảnh (chuyển thành tensor và chuẩn hóa)
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Đảm bảo kích thước ảnh là 100x100
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
])
########################     EXECUTE   ###########################

train_dataset = MyDataset(root_dir = DATA_ROOT_PATH/'train_images',transform=transform)
val_dataset = MyDataset(root_dir = DATA_ROOT_PATH/'val_images',transform=transform)

#Tạo data loader
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False,num_workers=4) #file test ko nên shuffle

# khởi tạo mô hình
model = MyModule()

# Chọn device: sử dụng GPU nếu có, không thì sử dụng CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Cài đặt loss function và optimizer
# criterion = nn.CrossEntropyLoss()
# Đếm số lượng mẫu theo lớp
label_counts = Counter(train_dataset.labels)
total = sum(label_counts.values())

# Tính weights theo tỷ lệ ngược lại
weights = [1.0 / label_counts[i] for i in range(len(label_counts))]
weights = torch.tensor(weights, dtype=torch.float).to(device)

# Tạo loss function có trọng số
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Train mô hình
def main():
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    log_path = "training_log.txt"
    # Ghi header cho file log
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
    for epoch in range(num_epochs):
        model.train()  # Đặt mô hình vào chế độ huấn luyện
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            # Tính loss
            loss = criterion(outputs, labels)
            # Backward pass và tối ưu hóa
            optimizer.zero_grad()  # Đặt gradient về 0
            loss.backward()  # Tính gradient
            optimizer.step()  # Cập nhật trọng số
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_dataloader)
        train_acc = 100 * correct / total
        

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_dataloader)
        val_acc = 100 * val_correct / val_total

        # === Scheduler step ===
        scheduler.step()

        # === Lưu lại lịch sử ===
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # === In ra ===
        print(f"Epoch [{epoch+1}/{num_epochs}] → "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        # === Ghi log vào file ===
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f}\n")

    # === Lưu ra file hoặc vẽ biểu đồ ===
    return train_losses, val_losses, train_accuracies, val_accuracies



if __name__ == '__main__':
    train_losses, val_losses, train_accs, val_accs = main()
    torch.save(model, OUTPUT_MODEL_PATH)
    print("✅ Đã lưu mô hình vào ecg_model.pth")
     # Vẽ biểu đồ
    import matplotlib.pyplot as plt

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()

