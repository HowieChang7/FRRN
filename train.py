# coding=utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical  # 数字标签转化成one-hot编码
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import LabelEncoder  # 只能对一维向量进行编码
#from model.frrn import check_print
from model import frrn
model = frrn.check_print()

# 设置随机数种子,保证实验可重复
seed = 7
np.random.seed(seed)  # np.random.random按顺序产生一组固定的数值，每次运行时可以使初始化的参数保持一致，避免初始化参数的不同对结果造成影响

img_w = 128
img_h = 128
batch_size = 1
# 有一个为背景
n_label = 2 + 1

classes = [0., 38., 75.]

labelencoder = LabelEncoder()
labelencoder.fit(classes)  # 将标签分配一个0 - n_classes-1之间的编码，将各种标签随机分配一个可数的连续编号

df_train = pd.read_csv('input2/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=1)  # 给定训练集和验证集

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

# data for training
def train_generator():  # 加载图片
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        end = len(ids_train_split)
        ids_train_batch = ids_train_split[0:end]
        for id in ids_train_batch.values:
            batch += 1
            img = cv2.imread('input2/train/{}.jpg'.format(id))
            label = cv2.imread('input2/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
            img = np.array(img, dtype="float") / 255.0  # 得到图像矩阵并进行归一化操作
            img = img_to_array(img)  # 将图片转为数组numpy.array格式
            train_data.append(img)
            #label = fix_img(label)
            label = img_to_array(label).reshape((img_w * img_h,))
            train_label.append(label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label).flatten()  # 转换为一维向量，LabelEncoder只支持1-Darray
                train_label = labelencoder.transform(train_label)  # 把图片像素转化为类别标签
                train_label = to_categorical(train_label, num_classes=n_label)  # 实现one-hot编码，把类别向量转换为二元类型矩阵
                train_label = train_label.reshape((batch_size, img_w * img_h, n_label))  # 转换遮罩的维度
                yield train_data, train_label
                train_data = []
                train_label = []
                batch = 0


def valid_generator():  # 加载图片
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        end = len(ids_valid_split)
        ids_valid_batch = ids_valid_split[0:end]
        for id in ids_valid_batch.values:
            batch += 1
            img = cv2.imread('input2/train/{}.jpg'.format(id))
            img = np.array(img, dtype="float") / 255.0
            img = img_to_array(img)  # 将图片转为数组numpy.array格式
            valid_data.append(img)
            label = cv2.imread('input2/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
            #label = fix_img(label)
            label = img_to_array(label).reshape((img_w * img_h,))
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label).flatten()  # 转换为一维向量，LabelEncoder只支持1-Darray
                valid_label = labelencoder.transform(valid_label)  # 把图片像素转化为类别标签
                valid_label = to_categorical(valid_label, num_classes=n_label)  # 实现one-hot编码，把类别向量转换为二元类型矩阵
                valid_label = valid_label.reshape((batch_size, img_w * img_h, n_label))  # 转换遮罩的维度
                yield valid_data, valid_label
                valid_data = []
                valid_label = []
                batch = 0

def train():
    EPOCHS = 50
    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4)
    modelcheck = ModelCheckpoint(filepath='weights/best_weights.hdf5', monitor='val_loss', save_best_only=True,
                                 save_weights_only=True)
    # tensorBoard = TensorBoard(log_dir='logs')
    callable = [ReduceLR, modelcheck]
    H = model.fit_generator(generator=train_generator(),
                            steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                            epochs=EPOCHS, verbose=2, validation_data=valid_generator(),
                            validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)),
                            callbacks=callable)
    #model.train_on_batch(x=train_generator(), y=valid_generator())
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Valid Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig("C://Users//asip//loss_figure.jpg")
    plt.show()

    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["dice_coeff0"], label="train_dice_coeff0")
    plt.plot(np.arange(0, N), H.history["val_dice_coeff0"], label="val_dice_coeff0")
    plt.plot(np.arange(0, N), H.history["dice_coeff1"], label="train_dice_coeff1")
    plt.plot(np.arange(0, N), H.history["val_dice_coeff1"], label="val_dice_coeff1")
    plt.plot(np.arange(0, N), H.history["dice_coeff2"], label="train_dice_coeff2")
    plt.plot(np.arange(0, N), H.history["val_dice_coeff2"], label="val_dice_coeff2")
    plt.title("Training dice_coeff and dice_coeff")
    plt.xlabel("Epoch #")
    plt.ylabel("dice_coeff")
    plt.legend(loc="lower right")
    plt.savefig("C://Users//asip//loss_figure1.jpg")
    plt.show()

    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["pixel_error"], label="train_error")
    plt.plot(np.arange(0, N), H.history["val_pixel_error"], label="val_error")
    plt.title("Training error and Valid error")
    plt.xlabel("Epoch #")
    plt.ylabel("pixel_error")
    plt.legend(loc="upper right")
    plt.savefig("C://Users//asip//loss_figure2 .jpg")
    plt.show()


if __name__ == '__main__':
    train()

