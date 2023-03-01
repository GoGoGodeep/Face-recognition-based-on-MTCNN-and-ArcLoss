import sys
from collections import Counter

import torch.cuda
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QPushButton, QGridLayout, QApplication
from torchvision import  transforms
from faceFeature import FaceNet
import os
from torch.nn import functional as F
import cv2 as cv
from mtcnndetect import Detector
from faceAlignment import imgAlign
from faceFeatureCompare import compare, feature

device = "cuda"

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

class FaceFeature(QWidget):

    def __init__(self):
        super().__init__()
        # 使用initUI()方法创建一个GUI
        self.initUI()

    # 窗口居中
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 人脸注册
    def faceSign(self):
        # 利用训练的特征提取网络进行人脸特征的注册
        net = FaceNet().cuda()
        net.load_state_dict(torch.load('param/facenet.pt'))
        net.eval()

        feature = {}
        for cls_name in os.listdir('featureimg'):

            for img_name in os.listdir(os.path.join('featureimg', cls_name)):
                person = transform(Image.open(os.path.join('featureimg', cls_name, img_name))).cuda()
                person_feature = net.encode(person[None, ...])

                # 特征归一化用于后面的余弦相似度对比
                facenorm = F.normalize(person_feature)
                feature[cls_name + '_' + img_name] = facenorm

        torch.save(feature, 'featurebase/feature.pt')
        self.txtlabel.setText("人脸注册成功！")


    # 识别人脸
    def faceDetect(self):

        # 相机读取
        url = "http://admin:admin@192.168.2.36:8081"
        cap = cv.VideoCapture(url)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        c = 0  # 帧数
        while True:
            # 逐帧捕获
            ret, frame = cap.read()
            # 如果正确读取帧，ret为True
            if not ret:
                break
            # 显示结果帧
            cv.imshow('frame', frame)

            detector = Detector()

            if (c + 20) % 30 == 0:  # 每几帧保存一次

                imgpath = 'outputs/' + str(c) + '.jpg'
                cv.imwrite(imgpath, frame)  # 存储视频帧为图像

                im = Image.open(imgpath).convert('RGB')

                boxes = detector.detect(im)
                for box in boxes:  # 多个框，每循环一次框一个人脸

                    lefteye_x = int(box[5])
                    lefteye_y = int(box[6])
                    righteye_x = int(box[7])
                    righteye_y = int(box[8])

                    imgAlign(imgpath, lefteye_x, lefteye_y,
                             righteye_x, righteye_y)

                im = Image.open(imgpath).convert('RGB')
                boxes = detector.detect(im)

                for box in boxes:  # 多个框，每循环一次框一个人脸
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    img = im.crop((x1, y1, x2, y2))
                    img.save(imgpath)

                    self.txtlabel.setText("正在识别！")

                    net = FaceNet().cuda()

                    # 使用
                    net.load_state_dict(torch.load('param/facenet.pt'))
                    net.eval()

                    person1 = transform(Image.open(imgpath)).cuda()
                    person1_feature = net.encode(person1[None, ...])

                    name = []
                    for key in feature:
                        cosa = compare(person1_feature, feature[key])

                        if cosa > 0.98:
                            name.append(key[:3])
                            # self.txtlabel.setText("识别成功！检测为{}".format(key[:3]))
                        else:
                            pass

                    occurence_count = Counter(name)
                    name_ = occurence_count.most_common(1)[0][0]

                    self.txtlabel.setText("识别成功！检测为{}".format(name_))
                    print("识别成功！检测为{}".format(name_))

            c += 1
            if cv.waitKey(1) == ord('q'):
                break

        # 完成所有操作后，释放捕获器
        cap.release()
        cv.destroyAllWindows()


    def initUI(self):
        # ————————————————————————窗口————————————————————————————
        self.resize(300, 200)
        self.center()  # 窗口居中

        # —————————————————————文字状态框—————————————————————————
        self.txtlabel = QtWidgets.QLabel()

        # ————————————————————————按钮————————————————————————————
        ClsButton = QPushButton("人脸识别")
        QuitButton = QPushButton("退出系统")
        QuitButton.clicked.connect(QCoreApplication.instance().quit)
        imgButton = QPushButton("人脸注册")

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self.txtlabel, 0, 1)
        grid.addWidget(imgButton, 1, 0)
        grid.addWidget(ClsButton, 1, 1)
        grid.addWidget(QuitButton, 1, 2)

        self.setLayout(grid)

        # ——————————————————上传图片并保存—————————————————————————
        imgButton.clicked.connect(self.faceSign)

        # ——————————————————调用模型进行推理————————————————————————
        ClsButton.clicked.connect(self.faceDetect)

        # ————————————————————————标题————————————————————————————
        self.setWindowTitle("人脸识别系统")
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceFeature()
    sys.exit(app.exec_())