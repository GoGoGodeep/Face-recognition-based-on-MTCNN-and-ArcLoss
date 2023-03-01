import torch
from faceFeature import FaceNet
from torchvision import transforms
from PIL import Image
from collections import Counter

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# 读取保存的特征向量文件
feature = torch.load('featurebase/feature.pt')

# 使用余弦相似度进行特征对比
def compare(face1, face2):
    # face1_norm = F.normalize(face1)
    # face2_norm = F.normalize(face2)
    cosa = torch.matmul(face1, face2.t())     # 矩阵乘法，进行转置

    return cosa

if __name__ == '__main__':

    net = FaceNet().cuda()
    net.load_state_dict(torch.load('param/facenet.pt'))
    net.eval()

    face1 = transform(Image.open('outputs/90.jpg')).cuda()
    face1_feature = net.encode(face1[None, ...])

    name = []
    for key in feature:
        cosa = compare(face1_feature, feature[key])

        if cosa > 0.98:
            name.append(key[:3])

    occurence_count = Counter(name)
    name_ = occurence_count.most_common(1)[0][0]

    print("检测为{}".format(name_))
