import torch
from faceFeature import FaceNet
from torchvision import transforms
import os
from torch.nn import functional as F
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# 利用训练的特征提取网络进行人脸特征的注册
net = FaceNet().cuda()
net.load_state_dict(torch.load('param/facenet.pt'))
net.eval()

def save_feature():

    feature = {}
    for cls_name in os.listdir('featureimg'):

        for img_name in os.listdir(os.path.join('featureimg', cls_name)):
            person = transform(Image.open(os.path.join('featureimg', cls_name, img_name))).cuda()
            person_feature = net.encode(person[None, ...])

            # 特征归一化用于后面的余弦相似度对比
            facenorm = F.normalize(person_feature)
            feature[cls_name + '_' + img_name] = facenorm

    torch.save(feature, 'featurebase/feature.pt')
    print("人脸数据注册成功！")


if __name__ == '__main__':

    feature = torch.load('featurebase/feature.pt')
    feature_list = []

    for key in feature:
        print(key, ":", feature[key])
