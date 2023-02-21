# 用来制作标签文件，将五官坐标与人脸四个坐标放进同个文件
bbox_label = r'E:\celeba\Anno\list_bbox_label.txt'
landmarks_label = r'E:\celeba\Anno\list_landmarks_celeba.txt'

bbox_ = []
with open(bbox_label) as bbox:
    bbox = bbox.readlines()
    for i in range(len(bbox)):
        bbox_.append(bbox[i].strip().split())

land_ = []
with open(landmarks_label) as land:
    land = land.readlines()
    for i in range(len(land)):
        land_.append(land[i].strip().split())

f = open(r'E:\celeba\label.txt', 'a')
for i in range(len(bbox)):
    f.write(' '.join(bbox_[i] + land_[i][1:]) + '\n')
#此处第一个引号内的内容为每一行中每一列之间的分隔符，第二个引号内的内容为隔行符
f.close()