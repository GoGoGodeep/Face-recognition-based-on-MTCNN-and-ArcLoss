import os

import cv2 as cv
from PIL import ImageFont, Image, ImageDraw
from torchvision import transforms
from faceAlignment import imgAlign
from facefeature import FaceNet, compare
from mtcnndetect import Detector
import torch
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# 字体
font = ImageFont.truetype("font/arial.ttf", size=23)

# 相机读取
url = "http://admin:admin@192.168.2.28:8081"
cap = cv.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

c = 0   # 帧数
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        break
    # 显示结果帧
    cv.imshow('frame', frame)

    detector = Detector()
    if c % 180 == 0:    # 每5帧保存一次

        imgpath = 'outputs/'+str(c)+'.jpg'
        cv.imwrite(imgpath, frame)  # 存储视频帧为图像

        # 检测人脸并进行人脸中心对齐
        print("-------------大小判断和中心对齐---------------")
        im = Image.open(imgpath).convert('RGB')
        boxes = detector.detect(im)
        for box in boxes:  # 多个框，每循环一次框一个人脸
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            lefteye_x = int(box[5])
            lefteye_y = int(box[6])
            righteye_x = int(box[7])
            righteye_y = int(box[8])
            nose_x = int(box[9])
            nose_y = int(box[10])
            leftmouth_x = int(box[11])
            leftmouth_y = int(box[12])
            rightmouth_x = int(box[13])
            rightmouth_y = int(box[14])

            # if abs(x1 - x2) > 400 or abs(x1 - x2) < 200:
            #     print("人脸尺寸不符合检测要求！")
            #     break
            # else:
            imgAlign(imgpath, lefteye_x, lefteye_y,
                     righteye_x, righteye_y)
            print("人脸中心对齐成功!")


        # 对中心对齐后的图片进行侦测
        print("-------------侦测---------------")
        im = Image.open(imgpath).convert('RGB')
        boxes = detector.detect(im)
        print("size:", im.size)
        # imDraw = ImageDraw.Draw(im)
        for box in boxes:  # 多个框，每循环一次框一个人脸
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            lefteye_x = int(box[5])
            lefteye_y = int(box[6])
            righteye_x = int(box[7])
            righteye_y = int(box[8])
            nose_x = int(box[9])
            nose_y = int(box[10])
            leftmouth_x = int(box[11])
            leftmouth_y = int(box[12])
            rightmouth_x = int(box[13])
            rightmouth_y = int(box[14])

            # print(x1, y1, x2, y2, '\n',
            #       lefteye_x, lefteye_y, righteye_x, righteye_y,
            #       nose_x, nose_y, leftmouth_x, leftmouth_y,
            #       rightmouth_x, rightmouth_y)
            #
            # print("conf:", box[4])  # 置信度
            # imDraw.rectangle((x1, y1, x2, y2), outline='red', width=5)
            # imDraw.point([(lefteye_x, lefteye_y),
            #               (righteye_x, righteye_y),
            #               (nose_x, nose_y),
            #               (leftmouth_x, leftmouth_y),
            #               (rightmouth_x, rightmouth_y)], fill='red')
            # imDraw.text((x1, y1), "{:.2f}".format(box[4]), font=font, fill=(255, 0, 255))

            img = im.crop((x1, y1, x2, y2))
            img.save(imgpath)

            net = FaceNet().cuda()

            # 使用
            net.load_state_dict(torch.load('param/facenet.pt'))
            net.eval()

            person1 = transform(Image.open(imgpath)).cuda()
            person1_feature = net.encode(person1[None, ...])

            for cls_name in os.listdir('facedata'):
                for img_name in os.listdir(os.path.join('facedata', cls_name))[:5]:
                    person2 = transform(Image.open(os.path.join('facedata', cls_name, img_name))).cuda()
                    person2_feature = net.encode(person2[None, ...])

                    cosa = compare(person1_feature, person2_feature)

                    if cosa > 0.98:
                        print("识别成功！为{}".format(cls_name))
                        break
                    else:
                        pass

    c += 1
    if cv.waitKey(1) == ord('q'):
        break

# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()

