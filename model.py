"""
卷积神经网络 模型类
"""
import tensorflow as tf


class ModelOfCNN(object):

    weights = {}
    biases = {}
    pass

    # Channels of img, default channels = 3, img in RGB
    def __init__(self, channels=3):

        # define weights's shape
        self.weights = {
            "w_conv1": self.weight_variable([5, 5, channels, 32]),
            "w_conv2": self.weight_variable([5, 5, 32, 64]),
            "w_fc1": self.weight_variable([7 * 7 * 64, 1024]),
            "w_fc2": self.weight_variable([1024, 10])
        }

        # define biases's shape
        self.biases = {
            "b_conv1": self.bias_variable([32]),
            "b_conv2": self.bias_variable([64]),
            "b_fc1": self.bias_variable([1024]),
            "b_fc2": self.bias_variable([10])
        }
    pass

    def weight_variable(self, shape):
        """
        给权值赋值，从正态分布片段中取值，标准差：0.01
        :param dtype: 数据类型
        :param shape: 卷积核属性
        :return:
        """
        initial = tf.truncated_normal(shape=shape, stddev=0.01, dtype="float")
        return tf.Variable(initial)

    # 偏置量函数
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype="float")
        return tf.Variable(initial)

    # 卷积操作
    def conv2d(self, x, w, b):
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # 池化操作
    def pooling(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding="SAME")

    # 局部归一化
    # def norm(self, x, lsize=4):
    #     return tf.nn.lrn(x, depth_radius=lsize, bias=1, alpha=0.001 / 9.0, beta=0.75)

    def output_cnn(self, images, keep_prob):
        """
        卷积神经网络模型
        :param images: 训练用的图片
        :return: a tensor  of shape [batch_size, NUM_CLASSES]
        """
        # 第一层
        hidden_conv1 = self.conv2d(images, self.weights["w_conv1"], self.biases["b_conv1"])
        hidden_pool1 = self.pooling(hidden_conv1)
        # hidden_norm1 = self.norm(hidden_pool1)

        # 第二层
        hidden_conv2 = self.conv2d(hidden_pool1, self.weights["w_conv2"], self.biases["b_conv2"])
        hidden_pool2 = self.pooling(hidden_conv2)
        # hidden_norm2 = self.norm(hidden_pool2)

        # 密集连接层
        hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, self.weights["w_fc1"].get_shape().as_list()[0]])
        hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, self.weights["w_fc1"]) + self.biases["b_fc1"])
        # 使用 Dropout 优化方法：用一个伯努利序列(0,1随机分布) * 神经元，随机选择每一次迭代的神经元
        hidden_fc1_dropout = tf.nn.dropout(hidden_fc1, keep_prob=keep_prob)

        # 输出层，没有做 softmax 回归
        logits = tf.add(tf.matmul(hidden_fc1_dropout, self.weights["w_fc2"]), self.biases["b_fc2"])
        return logits

    @staticmethod
    def output_w_b(self):
        print(self.weights)
        print(self.biases)
