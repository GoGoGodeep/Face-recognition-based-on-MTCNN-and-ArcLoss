import cv2
import math

def imgAlign(path, lefteye_x, lefteye_y, righteye_x, righteye_y):

    # 两只眼睛的中心点
    cx = (lefteye_x + righteye_x) / 2
    cy = (lefteye_y + righteye_y) / 2

    # 角度计算
    short1 = abs(lefteye_x - righteye_x)
    short2 = abs(lefteye_y - righteye_y)
    long = math.sqrt(short1 ** 2 + short2 ** 2)
    # print(long ** 2, short2 ** 2, short1 ** 2)

    angle = math.degrees(math.acos(
        (
                short2 * short2 - long * long - short1 * short1
        ) / (-2 * long * short1)))  # 求角度
    if abs(angle) > 10:
        src = cv2.imread(path)
        # 读取原图像的内容
        rows, cols, channel = src.shape
        # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        M = cv2.getRotationMatrix2D((cx, cy), -angle, scale=1)
        # 得到矩阵后得用到图像的仿射变换函数才可以进行最终图像的变化
        dst = cv2.warpAffine(src, M=M, dsize=(cols, rows))

        cv2.imwrite(path, dst)

        print("人脸中心对齐成功！")
    else:
        print("人脸中心已经对齐！")