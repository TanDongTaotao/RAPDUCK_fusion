import keras.backend as K
import tensorflow as tf


def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    # 确保输入类型一致
    ground_truth = tf.cast(ground_truth, tf.float32)
    predictions = tf.cast(predictions, tf.float32)

    # 获取batch_size
    batch_size = tf.shape(ground_truth)[0]

    # 展平并保持batch维度
    ground_truth = tf.reshape(ground_truth, [batch_size, -1])
    predictions = tf.reshape(predictions, [batch_size, -1])

    # 计算每个样本的dice系数
    intersection = tf.reduce_sum(predictions * ground_truth, axis=1)
    union = tf.reduce_sum(predictions, axis=1) + tf.reduce_sum(ground_truth, axis=1)
    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth))

    return 1 - dice