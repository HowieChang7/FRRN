# coding=utf-8
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical  # 数字标签转化成one-hot编码
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder  # 只能对一维向量进行编码

# img_w = 128
# img_h = 128
# batch_size = 583

img_w = 256
img_h = 256
batch_size = 600
# 有一个为背景
n_label = 2 + 1
classes = [0., 38., 75.]

labelencoder = LabelEncoder()
labelencoder.fit(classes)  # 将标签分配一个0 - n_classes-1之间的编码，将各种标签随机分配一个可数的连续编号

df_train = pd.read_csv('input1/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.0, random_state=1)  # 给定训练集和验证集

train_label = []
end = len(ids_train_split)
ids_train = ids_train_split[0:end]
for id in ids_train.values:
    label = cv2.imread('input1/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
    label = img_to_array(label).reshape((img_w * img_h,))
    train_label.append(label)
train_label = np.array(train_label).flatten()  # 转换为一维向量，LabelEncoder只支持1-Darray
train_label = labelencoder.transform(train_label)  # 把图片像素转化为类别标签
train_label = to_categorical(train_label, num_classes=n_label)  # 实现one-hot编码，把类别向量转换为二元类型矩阵
train_label = train_label.reshape((batch_size, img_w * img_h, n_label))  # 转换遮罩的维度

count = np.sum(train_label, axis=1)
count = np.sum(count, axis=0)  # 计算y_true中每一列元素的和
class_0 = count[0]  # 0类的像素个数
class_1 = count[1]  # 1类的像素个数
class_2 = count[2]  # 2类的像素个数

print(class_0)
print(class_1)
print(class_2)
weight1 = (class_0 + class_1 + class_2) / class_0
weight2 = (class_0 + class_1 + class_2) / class_1
weight3 = (class_0 + class_1 + class_2) / class_2
print(weight1)
print(weight2)
print(weight3)