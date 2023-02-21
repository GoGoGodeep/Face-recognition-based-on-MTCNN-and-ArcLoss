from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

class facedataset(Dataset):
    def __init__(self, dir):
        self.dataset = []
        for face_dir in os.listdir(dir):
            for face_filename in os.listdir(os.path.join(dir, face_dir)):
                self.dataset.append([os.path.join(dir, face_dir, face_filename), int(face_dir)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img_data = transform(Image.open(data[0]))

        return img_data, data[1]   # 返回图像数据与分类


if __name__ == '__main__':

    dataset = facedataset('facedata')

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for i in range(10):
        for xs, ys in dataloader:
            print(xs, ys)