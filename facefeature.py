import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import densenet201, DenseNet201_Weights
from torchvision.transforms import transforms
from faceDataset import facedataset

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# 与CenterLoss一样，我们把损失定义为网络
class ArcLoss(nn.Module):

    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn(feature_num, cls_num))

    # m为不同边之间的距离，根据情况进行修改，s即为分类数
    def forward(self, x, m=0.2, s=10):
        x_norm = F.normalize(x, dim=1)  # 在行上做标准化
        w_norm = F.normalize(self.w, dim=0)     # 在列上做标准化

        cos = torch.matmul(x_norm, w_norm) / 10    # /10是为了防止梯度爆炸
        a = torch.arccos(cos)  # 反求角度

        molecule = torch.exp(s * torch.cos(a + m))
        denominator = molecule + torch.sum(
            torch.exp(s * torch.cos(a)), dim=1, keepdim=True
        ) - torch.exp(s * torch.cos(a))    # 需要减去当前情况值

        arcsoftmax = torch.log(molecule / denominator)

        return arcsoftmax

class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 使用densenet网络，因为网络越深提取特征能力越强
        self.sub_net = nn.Sequential(
           densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        )
        # 特征提取
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 512, bias=False)    # 提取512个特征，不能太小
        )
        self.arc_softmax = ArcLoss(512, 6)  # 8个人的人脸

    def forward(self, x):
        y = self.sub_net(x)
        feature = self.feature_net(y)

        return feature, self.arc_softmax(feature, 1, 4)

    # 功能与上面一样，提取feature
    def encode(self, x):
        return self.feature_net(self.sub_net(x))


# 使用余弦相似度进行特征对比
def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    cosa = torch.matmul(face1_norm, face2_norm.t())     # 矩阵乘法，进行转置

    return cosa


if __name__ == '__main__':

    net = FaceNet().cuda()
    try:
        net.load_state_dict(torch.load("param/facenet.pt"))
        print("load success!")
    except:
        pass

    # 训练
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters())

    dataset = facedataset('facedata')
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    for epoch in range(5000):
        for xs, ys in dataloader:
            feature, cls = net(xs.cuda())

            loss = loss_fn(cls, ys.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(torch.argmax(cls, dim=1), ys)
            print("-", end=' ')

        print(str(epoch) + "  Loss ==> " + str(loss.item()))

        if (epoch+1) % 100 == 0:
            torch.save(net.state_dict(), "param/facenet.pt")
            print("save success！")


    # 使用
    # net.load_state_dict(torch.load('param/facenet.pt'))
    # net.eval()
    #
    # person1 = transform(Image.open('facedata/2/1.jpg')).cuda()
    # person1_feature = net.encode(person1[None, ...])
    #
    # person2 = transform(Image.open('facedata/1/3.jpg')).cuda()
    # person2_feature = net.encode(person2[None, ...])
    #
    # cosa = compare(person1_feature, person2_feature)
    # if cosa > 0.98:
    #     print("为同一人")
    # else:
    #     print("非同一人")