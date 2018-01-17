"""
训练与评估类
构建 TensorFlow 数据流图
"""
import model
import tensorflow as tf


class TrainingGraph(object):
    # placeholder for local data
    img_local_h = tf.placeholder("float", [None, 28, 28])
    lab_local_h = tf.placeholder("float", [None, 1])
    # channel of img ,  the value is 3 for RGB, 1 for gray
    channels = 3
    # keep_prob of dropout in model
    keep_prob = 1
    pass

    def __init__(self, channels, keep_prob):
        self.channels = channels
        self.keep_prob = keep_prob
        pass

    def get_loss(self, logits):
        """
        finish softmax
        :param logits: a tensor  of shape [batch_size, NUM_CLASSES]
        :return: float
        sparse_softmax_cross_entropy_with_logits ：
            softmax
            sparse_to_dense  [batchSize, one-hot:10]
            cross_entropy
        go to https://www.jianshu.com/p/fb119d0ff6a6 learn more
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.lab_local_h)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def build_graph(self):
        """
        logits: a tensor  of shape [batch_size, NUM_CLASSES]
        labels: a tensor of shape [batch_size]
        :return: graph of train_step and accuracy
        """
        # calculate the loss from model output
        cnn_model = model.ModelOfCNN(channels=self.channels)
        logits = cnn_model.output_cnn(images=self.img_local_h, keep_prob=self.keep_prob)
        loss = self.get_loss(logits)
        # build a train graph
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        # build a accuracy graph
        accuracy = tf.nn.in_top_k(logits, self.lab_local_h, 1)
        accuracy = tf.cast(accuracy, tf.float32)
        accuracy = tf.reduce_mean(accuracy)

        return train_step, accuracy
