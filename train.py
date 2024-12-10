import os
import tensorflow as tf
import albumentations as albu
import numpy as np
import gc
from keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
from RAPUNet.ModelArchitecture.DiceLoss import dice_metric_loss
from RAPUNet.ModelArchitecture import RAPUNet
from RAPUNet.CustomLayers import ImageLoader2D
from DUCKNet.ModelArchitecture import DUCK_Net
from JointModel.Joint import JointModel
import tensorflow_addons as tfa
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import os
import csv
from datetime import datetime


def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

img_size = 352
dataset_type = 'ISICDM'  # Options: kvasir/cvc-clinicdb/cvc-colondb/etis-laribpolypdb  # 仅用于文件存储命名
learning_rate = 1e-4
seed_value = 58800
rap_filters = 17
duck_filters = 17

starter_learning_rate = 1e-4
end_learning_rate = 1e-6
decay_steps = 1000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.2)
opts = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=learning_rate_fn)
b_size = 4

ct = datetime.now()


# rapunet
rap_model_type = "RAPUNet"
duck_model_type = "DuckNet"

model_type = "JointNet"

if not os.path.exists('ProgressFull'):
    os.makedirs('ProgressFull')

progress_path = 'ProgressFull/' + dataset_type + '_progress_csv_' + rap_model_type + str(ct) + '.csv'
progressfull_path = 'ProgressFull/' + dataset_type + '_progress_' + rap_model_type + str(ct) + '.txt'

model_path = 'ModelSave/' + dataset_type + '/' + rap_model_type + str(ct)

EPOCHS = 600
min_loss_for_saving = 0.1

rap_model = RAPUNet.create_model(img_height=img_size, img_width=img_size, input_chanels=3, out_classes=1,
                             starting_filters=rap_filters)
duck_model = DUCK_Net.create_model(img_height=img_size, img_width=img_size, input_chanels=3, out_classes=1, starting_filters=duck_filters)

# 创建联合模型
joint_model = JointModel(rap_model, duck_model, weight1=0.5, weight2=0.5)
# 编译联合模型
joint_model.compile(optimizer=opts,loss=dice_metric_loss)

# 创建一个示例输入来构建模型
dummy_input = tf.random.normal((1, img_size, img_size, 3))  # batch_size=1, height=img_size, width=img_size, channels=3
_ = joint_model(dummy_input)  # 这一步会构建模型
print("RAPUNet 模型结构:")
rap_model.summary()
print("CUDKNet 模型结构:")
duck_model.summary()
print("Joint 模型结构：")
joint_model.summary()


# data_path = "../data/ISICDM/train/" # Add the path to your data directory
# test_path = "../data/ISICDM/validation/" # Add the path to your data directory  # 估测试集
data_path = "D:/Projects_PyCharm/RAPDUCK_fusion/data/ISICDM/train/"  # Add the path to your data directory
test_path = "D:/Projects_PyCharm/RAPDUCK_fusion/data/ISICDM/validation/"  # Add the path to your data directory  # 估测试集

# X, Y = ImageLoader2D.load_data(img_size, img_size, -1, 'ISICDM', data_path, resize=True)

# split train/valid/test as 0.8/0.1/0.1
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle= True, random_state = seed_value)
# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.111, shuffle= True, random_state = seed_value)

# split train/vaid as 0.9/0.1 and fixed test data
# x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=seed_value)
# x_test, y_test = ImageLoader2D.load_data(img_size, img_size, -1, 'ISICDM', test_path, resize=True)
# 不划分版本
x_train, y_train = ImageLoader2D.load_data(img_size, img_size, -1, 'ISICDM', data_path, resize=True)

x_valid, y_valid = ImageLoader2D.load_data(img_size, img_size, -1, 'ISICDM', test_path, resize=True)


aug_train = albu.Compose([
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
    albu.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22),
                always_apply=True),
])


def augment_images():
    x_train_out = []
    y_train_out = []

    for i in range(len(x_train)):
        ug = aug_train(image=x_train[i], mask=y_train[i])
        x_train_out.append(ug['image'])
        y_train_out.append(ug['mask'])

    return np.array(x_train_out), np.array(y_train_out)


# 格式化日期时间字符串，避免使用不允许的字符
ct = datetime.now().strftime("%Y%m%d_%H%M%S")

# 更新文件路径
progress_path = f'ProgressFull/{dataset_type}_progress_csv_{model_type}_duck_filters_{duck_filters}_rap_filters_{rap_filters}_{ct}.csv'
progressfull_path = f'ProgressFull/{dataset_type}_progress_{model_type}_duck_filters_{duck_filters}_rap_filters_{rap_filters}_{ct}.txt'
plot_path = f'ProgressFull/{dataset_type}_progress_plot_{model_type}_duck_filters_{duck_filters}_rap_filters_{rap_filters}_{ct}.png'
model_path = f'ModelSaveTensorFlow/{dataset_type}/{model_type}_duck_filters_{duck_filters}_rap_filters_{rap_filters}_{ct}'

# 确保目录存在
os.makedirs(os.path.dirname(progress_path), exist_ok=True)
os.makedirs(os.path.dirname(model_path), exist_ok=True)


# 自定义日志记录器，继承自tf.keras.callbacks.Callback
class CustomLogger(tf.keras.callbacks.Callback):
    def __init__(self, filepath, separator=';'):
        super(CustomLogger, self).__init__()
        self.filepath = filepath
        self.separator = separator
        self.header_written = False

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.filepath, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=self.separator)

            # 写入表头
            if not self.header_written:
                writer.writerow(logs.keys())
                self.header_written = True

            # 写入数据
            writer.writerow(logs.values())


# 创建自定义logger
custom_logger = CustomLogger(progress_path)






for epoch in range(0, EPOCHS):

    print(f'Training, epoch {epoch}')
    # 获取当前实际的学习率
    current_lr = float(tf.keras.backend.get_value(joint_model.optimizer.lr))
    print('Current Learning Rate: ' + str(float(current_lr)))

    image_augmented, mask_augmented = augment_images()

    csv_logger = CSVLogger(progress_path, append=True, separator=';')

    joint_model.fit(x=image_augmented, y=mask_augmented, epochs=1, batch_size=b_size, validation_data=(x_valid, y_valid),
              verbose=1, callbacks=[custom_logger])

    prediction_valid = joint_model.predict(x_valid, verbose=0)
    loss_valid = dice_metric_loss(y_valid, prediction_valid)
    loss_valid = loss_valid.numpy()
    print("JointNet Loss Validation: " + str(loss_valid))

    # 无test版本
    # prediction_test = joint_model.predict(x_test, verbose=0)
    # loss_test = dice_metric_loss(y_test, prediction_test)
    # loss_test = loss_test.numpy()
    # print("JointNet Loss Test: " + str(loss_test))

    with open(progressfull_path, 'a') as f:
        f.write('epoch: ' + str(epoch) + '\nval_loss: ' + str(loss_valid) + '\n\n\n')
        # f.write('epoch: ' + str(epoch) + '\nval_loss: ' + str(loss_valid) + '\ntest_loss: ' + str(loss_test) + '\n\n\n')

    if min_loss_for_saving > loss_valid:
        min_loss_for_saving = loss_valid
        print("Saved JointNet model with val_loss: ", loss_valid)
        rap_model.save(model_path)
        rap_model.save("JointNet_best_model.h5")

    del image_augmented
    del mask_augmented

    gc.collect()

print("Loading the model")
model_path = "JointNet_best_model.h5"

joint_model = tf.keras.models.load_model(model_path, custom_objects={'dice_metric_loss': dice_metric_loss})



prediction_train = joint_model.predict(x_train, batch_size=1)[0]
prediction_valid = joint_model.predict(x_valid, batch_size=1)[0]

# prediction_test = model.predict(x_test, batch_size=1).numpy()

# prediction_train = prediction_train.reshape(prediction_train.shape[0], -1)
# prediction_valid = prediction_valid.reshape(prediction_valid.shape[0], -1)
# prediction_test = prediction_test.reshape(prediction_test.shape[0], -1)

print("Predictions done")

dice_train = f1_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                      np.ndarray.flatten(prediction_train > 0.5))
# dice_test = f1_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
#                      np.ndarray.flatten(prediction_test > 0.5))
dice_valid = f1_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                      np.ndarray.flatten(prediction_valid > 0.5))


print("Dice finished")

miou_train = jaccard_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                           np.ndarray.flatten(prediction_train > 0.5))
# miou_test = jaccard_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
#                           np.ndarray.flatten(prediction_test > 0.5))
miou_valid = jaccard_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                           np.ndarray.flatten(prediction_valid > 0.5))

print("Miou finished")

precision_train = precision_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                  np.ndarray.flatten(prediction_train > 0.5))
# precision_test = precision_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
#                                  np.ndarray.flatten(prediction_test > 0.5))
precision_valid = precision_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                  np.ndarray.flatten(prediction_valid > 0.5))

print("Precision finished")

recall_train = recall_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                            np.ndarray.flatten(prediction_train > 0.5))
# recall_test = recall_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
#                            np.ndarray.flatten(prediction_test > 0.5))
recall_valid = recall_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                            np.ndarray.flatten(prediction_valid > 0.5))

print("Recall finished")

accuracy_train = accuracy_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                np.ndarray.flatten(prediction_train > 0.5))
# accuracy_test = accuracy_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
#                                np.ndarray.flatten(prediction_test > 0.5))
accuracy_valid = accuracy_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                np.ndarray.flatten(prediction_valid > 0.5))

print("Accuracy finished")

final_file = 'results_' + model_type + '_' + dataset_type + '.txt'
print(final_file)

with open(final_file, 'a') as f:
    f.write(dataset_type + '\n\n')
    f.write(model_path + '\n\n')
    # f.write(' dice_train: ' + str(dice_train) + ' dice_valid: ' + str(dice_valid) + ' dice_test: ' + str(dice_test) + '\n\n')
    #
    # f.write(
    #     'miou_train: ' + str(miou_train) + ' miou_valid: ' + str(miou_valid) + ' miou_test: ' + str(miou_test) + '\n\n')
    # f.write('precision_train: ' + str(precision_train) + ' precision_valid: ' + str(
    #     precision_valid) + ' precision_test: ' + str(precision_test) + '\n\n')
    # f.write('recall_train: ' + str(recall_train) + ' recall_valid: ' + str(recall_valid) + ' recall_test: ' + str(
    #     recall_test) + '\n\n')
    # f.write(
    #     'accuracy_train: ' + str(accuracy_train) + ' accuracy_valid: ' + str(accuracy_valid) + ' accuracy_test: ' + str(
    #         accuracy_test) + '\n\n\n\n')
    f.write(' dice_train: ' + str(dice_train) + ' dice_valid: ' + str(dice_valid) + '\n\n')

    f.write(
        'miou_train: ' + str(miou_train) + ' miou_valid: ' + str(miou_valid) + '\n\n')
    f.write('precision_train: ' + str(precision_train) + ' precision_valid: ' + str(
        precision_valid) + '\n\n')
    f.write('recall_train: ' + str(recall_train) + ' recall_valid: ' + str(recall_valid) +  '\n\n')
    f.write(
        'accuracy_train: ' + str(accuracy_train) + ' accuracy_valid: ' + str(accuracy_valid) + '\n\n\n\n')

print('File done')



