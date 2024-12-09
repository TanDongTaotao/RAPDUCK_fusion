import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_math_ops import softmax
from RAPUNet.ModelArchitecture.DiceLoss import dice_metric_loss


class JointModel(tf.keras.Model):
    def __init__(self, model1, model2, weight1, weight2):
        super(JointModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.weight_algorithm = self.PDF
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

    def call(self, inputs):
        # 获取两个模型的预测
        pred1,feat1 = self.model1(inputs)
        pred2,feat2 = self.model2(inputs)
        feat1 = tf.stop_gradient(feat1)
        feat2 = tf.stop_gradient(feat2)

        try:
            # 计算权重
            weights = self.weight_algorithm(feat1, feat2)

            # 广播权重到预测形状
            w1 = tf.broadcast_to(weights[:,0], tf.shape(pred1))
            w2 = tf.broadcast_to(weights[:,1], tf.shape(pred2))

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
        trainable_vars = self.model1.trainable_variables + self.model2.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # 更新权重（注意：这里不需要重复调用apply_gradients）
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {'loss': loss}

    def predict_step(self, data):
        # 重写predict_step以确保正确处理预测
        x = data
        return self(x, training=False)

    # def predict_with_dynamic_weights(self, inputs, weight_calculator=None):
    #     """
    #     使用动态权重进行预测
    #
    #     Args:
    #         inputs: 输入数据
    #         weight_calculator: 动态权重计算函数，如果为None则使用默认权重
    #
    #     Returns:
    #         weighted_pred: 加权后的预测结果
    #         pred1: 模型1的预测结果
    #         pred2: 模型2的预测结果
    #         weights: 使用的权重
    #     """
    #     # 获取两个模型的预测和特征
    #     pred1, feat1 = self.model1(inputs)
    #     pred2, feat2 = self.model2(inputs)
    #
    #     if weight_calculator is not None:
    #         try:
    #             # 使用提供的权重计算函数
    #             weights = weight_calculator(pred1, pred2, feat1, feat2)
    #         except:
    #             # 如果计算失败，使用默认权重
    #             weights = [self.weight1, self.weight2]
    #             print("Warning: Dynamic weight calculation failed, using default weights")
    #     else:
    #         # 使用默认权重
    #         weights = [self.weight1, self.weight2]
    #
    #     # 加权组合预测结
    #     weighted_pred = weights[0] * pred1 + weights[1] * pred2
    #     return weighted_pred, pred1, pred2,weights


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
