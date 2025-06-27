import os
import numpy as np
import pywt
import wfdb
import scipy.signal as sg
from pathlib import Path
from glob import glob
import joblib
from tqdm import tqdm
import cv2
from tqdm import tqdm #tqdm is used to show a progress bar when processing many images
import concurrent.futures


###########################          GLOBAL VARIABLES           ################################################
ROOT_PATH = "E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/data/raw"
OUTPUT_PATH = "E:/ECG_Thanh/ECG-compact-model-/ECG_Classification/data/scalogram"
sampling_rate = 360
wavelet = "mexh"  # mexh, morl, gaus8, gaus4
scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

# List of all record files (.dat)
record_files = sorted(glob(os.path.join(ROOT_PATH, '*.dat')))

cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1  # for multi-process

############################          SUPPORT FUNCTIONS          ###############################################

def read_ecg_record(record):
    record_path = Path(ROOT_PATH)/ str(record) # Combine PATH and record using pathlib
    signal = wfdb.rdrecord(record_path.as_posix(), channels=[0]).p_signal[:, 0]
    annotation = wfdb.rdann(record_path.as_posix(), extension="atr")
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)
    return signal,r_peaks,labels

def median_filter(signal):
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    filtered_signal = signal - baseline 
    return filtered_signal

def remove_invalid_labels(r_peaks, labels):
    invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x', 'e', 'j', 'S', 'J', 'a', 'J', 'E', 'F', 'Q', 'f']
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
        #enumerate(labels) sẽ tạo ra một danh sách các cặp (index, label)
        #labels = ['N', '|', 'V', '~'] → enumerate(labels) sinh ra (0, 'N'), (1, '|'), (2, 'V'), (3, '~')
        #Vòng lặp for i, label in enumerate(labels)
        #Điều kiện if label not in invalid_labels
        #  ==> Danh sách kết quả indices Chứa danh sách các chỉ số i của label hợp lệ
    r_peaks, labels = r_peaks[indices], labels[indices]
    return r_peaks, labels

def align_r_peaks(r_peaks, filtered_signal,tol=0.05):
    newR = []
        #danh sách rỗng newR để lưu vị trí đỉnh R đã được căn chỉnh
    for r_peak in r_peaks: #Lặp qua từng điểm đỉnh R (r_peak) đã phát hiện trước đó.
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
            #Xác định giới hạn trái: Lùi lại một khoảng tol * sampling_rate từ vị trí r_peak để tạo một cửa sổ tìm kiếm.
            #np.maximum(..., 0) để đảm bảo không bị âm (tránh lỗi khi r_peak ở đầu tín hiệu).
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(filtered_signal))
            #Xác định giới hạn phải
        newR.append(r_left + np.argmax(filtered_signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int") # ép kiểu newR về int 

    #normalize signal 
    normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])
    return r_peaks, normalized_signal

# Nhóm các label thành 5 class
def AAMI_categories(labels): 
    AAMI = {
        "N": 0,   # Normal
        "L": 1,   # LBBB
        "R": 2,  # RBBB
        "V": 3,  # PVC
        "/": 4 # PB
    }
    categories = [AAMI[label] for label in labels]
        #Nếu label = "N", thì AAMI["N"] sẽ trả về 0
        #[AAMI[label] for label in labels] là list comprehension trong Python, nó tạo ra một list mới. 
        #Kết quả là một danh sách các giá trị mã phân loại tương ứng với từng nhãn trong labels.
    return categories

def segmentation(normalize_signal, r_peaks, categories):
    before, after = 90, 110 ## Lấy đoạn tín hiệu 200ms quanh R-peak
    beats, beat_labels = [], []
    for r, category in zip(r_peaks, categories):
        start = r - before
        end = r + after
        if start >= 0 and end < len(normalize_signal): # nếu không đủ dữ liệu (gần biên), bỏ qua
            beat = normalize_signal[start:end]
            beats.append(beat)
            beat_labels.append(category)
        
    return beats, beat_labels

def CWT(record_name, beats, beat_labels, image_size = 100):

    os.makedirs(OUTPUT_PATH, exist_ok=True) #Tạo thư mục gốc để lưu ảnh (nếu chưa tồn tại). Dùng exist_ok=True để tránh lỗi nếu thư mục đã có
    
    for i, (beat, beat_label) in enumerate(tqdm(zip(beats, beat_labels), total=len(beats))):
            #zip(beats, beat_labels)	Gộp hai list beats và beat_labels lại thành các cặp (beat, label)
            #tqdm(...)	Hiển thị progress bar

        # Chuyển CWT
        # 1. Tính CWT → scalogram
        coef, _ = pywt.cwt(beat, scales, wavelet)
        scalogram = np.abs(coef)
        # 2. Chuẩn hóa về [0, 255] để lưu ảnh
        scalogram -= np.min(scalogram)
        scalogram /= np.max(scalogram) + 1e-6 #+1e-6: tránh chia cho 0
        scalogram *= 255
        scalogram = scalogram.astype(np.uint8) #Đổi sang uint8: đúng định dạng ảnh grayscale.
        # 3. Resize ảnh
        resized = cv2.resize(scalogram, (image_size, image_size), interpolation=cv2.INTER_CUBIC) #dùng nội suy INTER_CUBIC để giữ chất lượng cao
        # 4. Tạo thư mục theo label
        label_dir = os.path.join(OUTPUT_PATH, str(beat_label))
        os.makedirs(label_dir, exist_ok=True)
        # 5. Lưu ảnh dưới dạng .png (grayscale)
        save_path = os.path.join(label_dir, f'{record_name}_{beat_label}_{i}.png')
        cv2.imwrite(save_path, resized)  # ảnh grayscale


##############################              PROCESS 1 RECORD                   ####################################
# Định nghĩa mapping class và record tương ứng
class_map = {
    0: {  # N
        'records': {'102', '104', '105', '106', '114', '116'},
        'ann': ['N']
    },
    1: {  # LBBB
        'records': {'109', '111', '207', '214'},
        'ann': ['L']
    },
    2: {  # RBBB
        'records': {'118', '124', '212', '231', '232', '207'},
        'ann': ['R']
    },
    3: {  # PVC
        'records': {'105', '106', '107', '114', '116', '119', '203', '207', '208', '210', '213', '214', '215', '221', '228', '233'},
        'ann': ['V']
    },
    4: {  # PB
        'records': {'102', '104', '107', '217'},
        'ann': ['/']
    }
}

def extract_beats_and_labels(record_name, signal, r_peaks, labels):
    """
    Trích xuất các beat và nhãn theo class_map, chỉ lấy đúng annotation và record tương ứng.
    """
    before, after = 90, 110
    beats, beat_labels = [], []
    for class_idx, info in class_map.items():
        if record_name in info['records']:
            for r, label in zip(r_peaks, labels):
                if label in info['ann']:
                    start = r - before
                    end = r + after
                    if start >= 0 and end < len(signal):
                        beat = signal[start:end]
                        beats.append(beat)
                        beat_labels.append(class_idx)
    return beats, beat_labels

def process_record(record_name):
    print(f'\nProcessing record: {record_name}')
    # 1. Load dữ liệu
    signal, r_peaks, labels = read_ecg_record(record_name)
    # 2. Lọc tín hiệu
    filtered_signal = median_filter(signal)
    # 3. lọc nhãn invalid
    r_peaks, labels = remove_invalid_labels(r_peaks, labels)
    # 4. căn chỉnh R, chuẩn hóa
    r_peaks, normalize_signal = align_r_peaks(r_peaks, filtered_signal)
    # 5. segmentation theo class_map
    beats, beat_labels = extract_beats_and_labels(str(record_name), normalize_signal, r_peaks, labels)
    # 6. Chuyển thành scalogram và lưu
    CWT(record_name, beats, beat_labels)
    print(f'done: {record_name}')

def main():
    all_records = [
            '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
            '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
            '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
            '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
            '231', '232', '233', '234'
        ]
    # Sử dụng ProcessPoolExecutor để chạy song song
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_record, all_records)

if __name__ == "__main__":
    main()