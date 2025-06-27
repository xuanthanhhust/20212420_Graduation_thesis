import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms


from Class_Model import MyModule  # hoặc resnet18 nếu không custom
from Class_Dataset import MyDataset
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
# ĐĂNG KÝ LỚP QUANTIZED CHO STOCHASTICDEPTH
from aimet_torch.nn import QuantizationMixin
from torchvision.ops import StochasticDepth

@QuantizationMixin.implements(StochasticDepth)
class QuantizedStochasticDepth(QuantizationMixin, StochasticDepth):
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, input):
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)
        with self._patch_quantized_parameters():
            ret = super().forward(input)
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)
        return ret


def plot_weight_range_per_channel(layer, title="Weight Range per Output Channel", save_path = None):
    """
    Vẽ biểu đồ boxplot cho từng output channel.
    Áp dụng cho Conv2d hoặc Depthwise Conv2d (shape: [out_ch, in_ch, kH, kW])
    """
    if not isinstance(layer, torch.nn.Conv2d):
        print("⚠️ Lớp không phải Conv2d.")
        return

    # Lấy trọng số và chuyển về numpy
    weight = layer.weight.data.cpu().numpy()

    # Shape: [out_channels, in_channels, kH, kW]
    out_channels = weight.shape[0]
    
    # Gom các giá trị kernel lại theo từng output channel
    boxplot_data = []
    for c in range(out_channels):
        # Trường hợp Conv thường: lấy toàn bộ kernel của output channel c
        # Flatten tất cả giá trị (bao gồm in_channels * kernel_h * kernel_w)
        w = weight[c].flatten()
        boxplot_data.append(w)

    # Vẽ boxplot
    plt.figure(figsize=(12, 5))
    plt.boxplot(boxplot_data, showfliers=True, patch_artist=True)
    plt.title(title)
    plt.xlabel("Output Channel Index")
    plt.ylabel("Weight")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📷 Saved plot to {save_path}")
    plt.show()



# ====== CẤU HÌNH ======
DATA_ROOT_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/data")
MODEL_PATH = Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/model_save/ecg_model_compact.pth")
QAT_MODEL_EXPORT_PATH =Path("E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/model_save/ecg_model_quantized.onnx")
BATCH_SIZE = 128
NUM_EPOCHS_QAT = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
# ====== DỮ LIỆU ======
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])

    train_dataset = MyDataset(DATA_ROOT_PATH / 'train_images', transform=transform)
    val_dataset = MyDataset(DATA_ROOT_PATH / 'test_images', transform=transform)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4)

# ====== LOAD MODEL PRUNE (.pth) ======
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval().to(device)



# ===== Trước khi CLE =====
    print("🔍 Trước CLE — min-max theo channel:")
    plot_weight_range_per_channel(model.model.features[1][0].block[0][0], title="Before CLE",  save_path="before_cle_weights.png")
    
# ====== CROSS-LAYER EQUALIZATION (CLE) ======
    equalize_model(model, dummy_input=torch.randn(1, 1, 100, 100).to(device))

# ===== Sau khi CLE =====
    print("📉 Sau CLE — min-max theo channel:")
    plot_weight_range_per_channel(model.model.features[1][0].block[0][0], title="After CLE", save_path="after_cle_weights.png")

# ====== OPTIONAL: BN Folding + Equalization (nếu model hỗ trợ) ======
    fold_all_batch_norms(model, input_shapes=[(1, 1, 100, 100)])

# ====== QUANTIZATION SIMULATION (QAT) ======
    dummy_input = torch.randn(1, 1, 100, 100).to(device)
    sim = QuantizationSimModel(model, dummy_input=dummy_input, default_output_bw=8, default_param_bw=8, quant_scheme='tf_enhanced')
    sim.compute_encodings(lambda m, i: m(i), dummy_input)

# ====== QAT FINE-TUNING ======
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(sim.model.parameters(), lr=1e-4)

    log_path = "qat_training_log.txt"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    def evaluate(model, loader):
        model.eval()
        total, correct, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = running_loss / len(loader)
        acc = 100 * correct / total
        return avg_loss, acc

    print("🔥 Bắt đầu QAT training...")
    sim.model.train()
    for epoch in range(NUM_EPOCHS_QAT):
        running_loss = 0.0
        correct, total = 0, 0
        sim.model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_QAT}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = sim.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc = evaluate(sim.model, val_loader)

        print(f"✅ Epoch {epoch+1} — Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f}\n")

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

# ...existing code...

    # ====== TEST MÔ HÌNH QAT TRƯỚC KHI LƯU ======
    sim.model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = sim.model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính confusion matrix, precision, recall, f1
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Classification Report:\n", classification_report(all_labels, all_preds))

    # ====== EXPORT MODEL QAT sang ONNX (nếu muốn) ======
    sim.export(path = './', filename_prefix= "ecg_model_quantized", dummy_input = torch.randn(1, 1, 100, 100) )
    print("✅ Đã export mô hình QAT sang ONNX:", QAT_MODEL_EXPORT_PATH)
    # import os 
    #     # ====== ĐO KÍCH THƯỚC FILE THỰC TẾ ======
    # size_before_pth = os.path.getsize("model_before_qat.pth") / 1024 / 1024  # MB
    # size_after_pth = os.path.getsize("model_after_qat.pth") / 1024 / 1024    # MB
    # size_after_onnx = os.path.getsize(QAT_MODEL_EXPORT_PATH) / 1024 / 1024   # MB

    # # ====== SO SÁNH FLOPs & PARAMETER GIỮA 2 FILE .pth ======
    # from thop import profile

    # model_before = torch.load("ecg_model_compact.pth", map_location=device)
    # model_after = torch.load("model_after_qat.pth", map_location=device)
    # dummy = torch.randn(1, 1, 100, 100).to(device)

    # flops_before, params_before = profile(model_before, inputs=(dummy,), verbose=False)
    # flops_after, params_after = profile(model_after, inputs=(dummy,), verbose=False)



    # print("\n=== So sánh kích thước file model ===")
    # print(f"File .pth trước quantize: {size_before_pth:.2f}MB")
    # print(f"File .pth sau quantize:   {size_after_pth:.2f}MB")
    # print(f"File .onnx sau quantize:  {size_after_onnx:.2f}MB")
    # print(f"ONNX nhỏ hơn .pth (sau quantize): {(size_after_pth-size_after_onnx)/size_after_pth*100:.2f}%")

    # print("\n=== So sánh FLOPs & Parameters giữa 2 file .pth ===")
    # print(f"FLOPs trước: {flops_before/1e6:.2f}M | sau: {flops_after/1e6:.2f}M | giảm: {(flops_before-flops_after)/flops_before*100:.2f}%")
    # print(f"Params trước: {params_before/1e3:.2f}K | sau: {params_after/1e3:.2f}K | giảm: {(params_before-params_after)/params_before*100:.2f}%")