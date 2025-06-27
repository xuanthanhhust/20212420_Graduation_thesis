from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image

class MyDataset(Dataset): #class quản lí scalogram images
    def __init__(self, root_dir, transform = None):
        # Args:
        # root_dir (string): Đường dẫn đến thư mục chứa ảnh (ví dụ: 'data/train' hoặc 'data/test')
        self.root_dir = root_dir

        # Lưu danh sách tất cả đường dẫn ảnh và label tương ứng
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Duyệt qua từng thư mục label (0, 1, 2, 3)
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label) #tạo đg dẫn tới thư mục từng label
            if os.path.isdir(label_dir): #kiểm tra xem label_dir có phải là một thư mục hay không (đúng thì TRUE)
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(int(label))  # label là tên thư mục
    
    def __len__(self):
        #mỗi Dataset cần phải định nghĩa phương thức __len__() để DataLoader biết có bao nhiêu sample.
        #self.image_paths là một danh sách chứa đường dẫn đến tất cả các ảnh ECG bạn đã load vào trong __init__()
        return len(self.image_paths)    #trả về số lượng mẫu (số ảnh ECG) trong dataset
    
    def __getitem__(self, idx):
        # Mở ảnh
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # nếu muốn convert RGB thì image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
        #Trả về một mẫu dữ liệu tại chỉ số idx
        #Được gọi tự động bởi DataLoader trong vòng lặp train/test
        #return:
            #img: Tensor có shape [1, 100, 100]  # nếu dùng .convert('L') + transform.ToTensor()
            #label:                             # label tương ứng với folder chứa ảnh