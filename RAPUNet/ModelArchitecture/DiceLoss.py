import tensorflow as tf

def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = tf.cast(ground_truth, tf.float32)
    predictions = tf.cast(predictions, tf.float32)
    ground_truth = tf.keras.layers.Flatten()(ground_truth)
    predictions = tf.keras.layers.Flatten()(predictions)
    intersection = tf.reduce_sum(predictions * ground_truth)
    union = tf.reduce_sum(predictions) + tf.reduce_sum(ground_truth)
    dice = (2. * intersection + smooth) / (union + smooth)
    return (1 - dice)

