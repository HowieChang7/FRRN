from keras.losses import categorical_crossentropy
import keras.backend as K
import tensorflow as tf
from functools import partial
from itertools import product
import numpy as np

nb_classes = 3

def mycrossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def pixelwise_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)  # the class probas of each sample sum to 1
    # count = K.sum(y_true, axis=1)
    # count = K.sum(count, axis=0)  # 计算y_true中每一列元素的和
    # class_0 = count[0]  # 0类的像素个数
    # class_1 = count[1]  # 1类的像素个数
    # class_2 = count[2]  # 2类的像素个数
    # weight1 = (class_0 + class_1 + class_2) / (class_0+1e-7)
    # weight2 = (class_0 + class_1 + class_2) / (class_1+1e-7)
    # weight3 = (class_0 + class_1 + class_2) / (class_2+1e-7)
    # weight1 = 2.18913339225
    # weight2 = 3.29803982906
    # weight3 = 4.16687802339
    weight1 = 2.42766327038
    weight2 = 1.76075517024
    weight3 = 49.6447238846
    # Get cross-entropy losses for each pixel.
    pixel_losses0 = - K.mean(weight1 * y_true[:, :, 0] * K.log(y_pred[:, :, 0]))
    pixel_losses1 = - K.mean(weight2 * y_true[:, :, 1] * K.log(y_pred[:, :, 1]))
    pixel_losses2 = - K.mean(weight3 * y_true[:, :, 2] * K.log(y_pred[:, :, 2]))
    pixel_losses = pixel_losses0 + pixel_losses1 + pixel_losses2  # 每一类的交叉熵损失求和
    return pixel_losses


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


# def dice_coeff(y_true, y_pred):
#     smooth = 1.
#     y_true = concatenate([y_true, 1. - y_true], axis=0)
#     y_true_f = K.flatten(y_true)
#     y_pred = K.cast(K.greater(y_pred, 0.5), 'float32')
#     y_pred = concatenate([y_pred, 1 - y_pred], axis=0)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return score


def dice_coeff0(y_true, y_pred):
    dice0 = dice_coeff(y_true[:, :, 0], y_pred[:, :, 0])
    return dice0

def dice_coeff1(y_true, y_pred):
    dice1 = dice_coeff(y_true[:, :, 1], y_pred[:, :, 1])
    return dice1

def dice_coeff2(y_true, y_pred):
    dice2 = dice_coeff(y_true[:, :, 2], y_pred[:, :, 2])
    return dice2


def dice_loss(y_true, y_pred):
    dice0 = dice_coeff(y_true[:, :, 0], y_pred[:, :, 0])
    dice1 = dice_coeff(y_true[:, :, 1], y_pred[:, :, 1])
    dice2 = dice_coeff(y_true[:, :, 2], y_pred[:, :, 2])
    dice = dice0 + dice1 + dice2
    loss = 3 - dice
    return loss


def pixel_error(y_true, y_pred):
    true = K.argmax(y_true, axis=-1)
    pred = K.argmax(y_pred, axis=-1)
    error_num = tf.cast(tf.not_equal(pred, true), tf.float32)
    error = tf.reduce_mean(error_num, name='pixel_error')
    return error

# the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : jaccard coefficient
# alpha+beta=1   : produces set of F*-scores
def dice_coeff_multi(y_true, y_pred):
    alpha = 0.5
    beta = 0.5
    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # 类别i的概率
    p1 = ones - y_pred  # 不是类别i的概率
    g0 = y_true
    g1 = ones - y_true
    num = K.sum(p0 * g0, (0, 1))
    den = num + alpha * K.sum(p0 * g1, (0, 1)) + beta * K.sum(p1 * g0, (0, 1))
    T = K.sum(num / den)
    return T


def categorical_crossentropy_loss(y_true, y_pred):
    k = 128*64
    top = 128 * 128
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.sum(y_pred * y_true, axis=2)
    y_pred_f = K.flatten(y_pred)
    y_pred_f, _ = tf.nn.top_k(y_pred_f, top)
    # y_pred_top_k, y_pred_ind_k = tf.nn.top_k(y_pred_f, top, sorted=False)
    # y_true_top_k = getitems_by_indices(y_true_f, y_pred_ind_k)
    # y_pred_f = tf.gather(y_pred_f, tf.nn.top_k(y_pred_f, k=128*128).indices)
    loss = 0.
    for i in range(k, top):
        loss += -K.log(y_pred_f[i])
    loss /= k
    return loss


def seg_score(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    true_sum = K.sum(y_true_f); pred_sum = K.sum(y_pred_f)
    if(true_sum > pred_sum):
        max_sum = true_sum
    else:
        max_sum = pred_sum
    return (intersection + smooth) / (max_sum + smooth)

def seg_score_loss(y_true, y_pred):
    return 1-seg_score(y_true, y_pred)
