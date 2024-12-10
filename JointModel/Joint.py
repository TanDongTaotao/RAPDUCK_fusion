import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_math_ops import softmax
from RAPUNet.ModelArchitecture.DiceLoss import dice_metric_loss


class JointModel(tf.keras.Model):
    def __init__(self, model1, model2, weight1, weight2):
        super(JointModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.train_weight_algorithm = self.PDF
        self.predict_weight_algorithm = self.predict_PDF
        self.weight1 = weight1
        self.weight2 = weight2

        self.ConfidNet_model1 = tf.keras.Sequential([
            tf.keras.layers.Dense(7744 * 2, input_shape=(7744,)),
            tf.keras.layers.Dense(7744),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
        ])
        self.ConfidNet_model2 = tf.keras.Sequential([
            tf.keras.layers.Dense(7744 * 2, input_shape=(7744,)),
            tf.keras.layers.Dense(7744),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, inputs, training=None, mask=None):
        # 获取两个模型的预测
        pred1,feat1 = self.model1(inputs)
        pred2,feat2 = self.model2(inputs)
        feat1 = tf.stop_gradient(feat1)
        feat2 = tf.stop_gradient(feat2)

        # print("ConfidNet_model1 可训练参数:")
        # for layer in self.ConfidNet_model1.layers:
        #     print(f"{layer.name}: trainable = {layer.trainable}")
        #
        # print("ConfidNet_model2可训练参数: ")
        # for layer in self.ConfidNet_model2.layers:
        #     print(f"{layer.name}: trainable = {layer.trainable}")

        try:

            if training:
                # 计算权重
                weights = self.train_weight_algorithm(feat1, feat2)
            else:
                weights = self.predict_weight_algorithm(feat1, feat2,pred1,pred2)

            # 广播权重到预测形状
            w1 = tf.reshape(weights[:, 0], [-1, 1, 1, 1])  # shape: [batch_size, 1, 1, 1]
            w2 = tf.reshape(weights[:, 1], [-1, 1, 1, 1])  # shape: [batch_size, 1, 1, 1]
            # w1 = tf.broadcast_to(weights[:,0], tf.shape(pred1))
            # w2 = tf.broadcast_to(weights[:,1], tf.shape(pred2))

        except Exception as e:
            tf.print("Exception in weight calculation:", e)
            # 使用默认权重
            w1 = tf.ones_like(pred1) * self.weight1
            w2 = tf.ones_like(pred2) * self.weight2

        # 加权组合预测结果
        weighted_pred = tf.stop_gradient(w1) * pred1 + tf.stop_gradient(w2) * pred2

        return weighted_pred

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            weighted_pred = self(x, training=True)
            loss = self.compiled_loss(y, weighted_pred)

        # 计算梯度
            # 计算梯度：包含所有可训练变量
            trainable_vars = (self.model1.trainable_variables +
                              self.model2.trainable_variables +
                              self.ConfidNet_model1.trainable_variables +
                              self.ConfidNet_model2.trainable_variables)
        gradients = tape.gradient(loss, trainable_vars)

        # 更新权重（注意：这里不需要重复调用apply_gradients）
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {'loss': loss}

    def predict_step(self, data):
        """
        重写predict_step以确保正确处理预测
        """
        x = data
        pred = self(x, training=False)
        return pred


    def PDF(self,feat1, feat2):
        """
        计算动态权重
        Args:
            feat1, feat2: 预测结果 [batch_size, height, width, channels]
        Returns:
            weights: shape [batch_size, 2]
        """
        # 计算每个预测的平均值,作为mono
        feat1_mean = self.ConfidNet_model1(feat1)  # [batch_size]
        feat2_mean = self.ConfidNet_model2(feat2)  # [batch_size]

        # 计算holo特征
        feat1_holo = tf.math.log(feat1_mean + 1e-8) / (tf.math.log(feat1_mean * feat2_mean + 1e-8) + 1e-8)
        feat2_holo = tf.math.log(feat2_mean + 1e-8) / (tf.math.log(feat1_mean * feat2_mean + 1e-8) + 1e-8)

        cb_feat1 = tf.stop_gradient(feat1_mean) + tf.stop_gradient(feat1_holo)
        cb_feat2 = tf.stop_gradient(feat2_mean) + tf.stop_gradient(feat2_holo)


        # 堆叠并应用softmax
        w_all = tf.stack([cb_feat1, cb_feat2], axis=1)  # [batch_size, 2]
        w_all = tf.nn.softmax(w_all, axis=1)

        return w_all


    def predict_PDF(self,feat1, feat2,pred1,pred2):
        """
        预测时的PDF权重
        :param x:
        :return:
        """
        # 计算每个预测的平均值,作为mono
        feat1_mean = self.ConfidNet_model1(feat1)  # [batch_size]
        feat2_mean = self.ConfidNet_model2(feat2)  # [batch_size]

        # 计算holo特征
        feat1_holo = tf.math.log(feat1_mean + 1e-8) / (tf.math.log(feat1_mean * feat2_mean + 1e-8) + 1e-8)
        feat2_holo = tf.math.log(feat2_mean + 1e-8) / (tf.math.log(feat1_mean * feat2_mean + 1e-8) + 1e-8)

        cb_feat1 = feat1_mean + feat1_holo
        cb_feat2 = feat2_mean + feat2_holo

        # 计算du

        target_value = 1 / tf.cast(tf.shape(pred1)[1], tf.float32)
        feat1_du = tf.reduce_mean(tf.abs(pred1 - target_value), axis=[1, 2, 3], keepdims=True)  # [batch_size, 1, 1, 1]
        feat2_du = tf.reduce_mean(tf.abs(pred2 - target_value), axis=[1, 2, 3], keepdims=True)  # [batch_size, 1, 1, 1]

        # 调整维度以匹配cb_feat
        feat1_du = tf.squeeze(feat1_du, axis=[1, 2, 3])  # [batch_size]
        feat2_du = tf.squeeze(feat2_du, axis=[1, 2, 3])  # [batch_size]
        feat1_du = tf.expand_dims(feat1_du, axis=1)  # [batch_size, 1]
        feat2_du = tf.expand_dims(feat2_du, axis=1)  # [batch_size, 1]

        # 判断DU
        condition = feat1_du > feat2_du

        # RC校准
        rc_feat1 = tf.where(condition,tf.ones_like(feat1_du),feat1_du / feat2_du)
        rc_feat2 = tf.where(condition,feat2_du / feat1_du,tf.ones_like(feat2_du))
        ccb_feat1 = cb_feat1 * rc_feat1
        ccb_feat2 = cb_feat2 * rc_feat2

        # 堆叠并应用softmax
        w_all = tf.stack([ccb_feat1, ccb_feat2], axis=1)  # [batch_size, 2]
        w_all = tf.nn.softmax(w_all, axis=1)
        return w_all