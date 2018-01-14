import os
import numpy as np
import tensorflow as tf


def weight_variable(shape, dtype):
    """
    给权值赋值，从正态分布片段中取值，标准差：0.01
    :param shape: 卷积核属性
    :return:
    """
    initial = tf.truncated_normal(shape=shape, stddev=0.1, dtype=dtype)
    return tf.Variable(initial)


# 偏置量函数
def bias_variable(shape, dtype):
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.Variable(initial)

# 定义权值 shape
weights = {
    "w_conv1": weight_variable([5, 5, 3, 32], dtype="float"),
    "w_conv2": weight_variable([5, 5, 32, 64], dtype="float"),
    "w_fc1": weight_variable([7 * 7 * 64, 1024], dtype="float"),
    "w_fc2": weight_variable([1024, 10], dtype="float")
    }

# 定义偏置量 shape
biases = {
    "b_conv1": bias_variable([32], dtype="float"),
    "b_conv2": bias_variable([64], dtype="float"),
    "b_fc1": bias_variable([1024], dtype="float"),
    "b_fc2": bias_variable([10], dtype="float")
    }


# 卷积操作
def conv2d(x, w, b):
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# 池化操作
def pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


# 局部相应归一化
def norm(x, lsize=4):
    return tf.nn.lrn(x, depth_radius=lsize, bias=1, alpha=0.001/9.0, beta=0.75)


# cnn 模型
def cnn_model(images):
    # 第一层
    hidden_conv1 = conv2d(images, weights["w_conv1"], biases["b_conv1"])
    hidden_pool1 = pooling(hidden_conv1)
    # hidden_norm1 = norm(hidden_pool1)

    # 第二层
    hidden_conv2 = conv2d(hidden_pool1, weights["w_conv2"], biases["b_conv2"])
    hidden_pool2 = pooling(hidden_conv2)
    # hidden_norm2 = norm(hidden_pool2)

    # 密集连接层
    hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, weights["w_fc1"].get_shape().as_list()[0]])
    hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, weights["w_fc1"])+biases["b_fc1"])
    # 使用 Dropout 优化方法：用一个伯努利序列(0,1随机分布) * 神经元，随机选择每一次迭代的神经元
    hidden_fc1_dropout = tf.nn.dropout(hidden_fc1, 0.5)

    # 输出层，没有做 softmax 回归
    logits = tf.add(tf.matmul(hidden_fc1_dropout, weights["w_fc2"]), biases["b_fc2"])
    return logits


def get_loss(logits, labels):
    """
    损失函数，包含输出层的 softmax 回归
    :param logits:
    :param labels:
    :return: loss
    """
    '''
    sparse_softmax_cross_entropy_with_logits 函数执行了三个计算步骤：
    softmax 回归
    sparse_to_dense 简单说就是把标签变成适用于神经网络输出的形式，也就 batchsize 的 onehot 行向量
    cross_entropy 交叉熵
    前往 https://www.jianshu.com/p/fb119d0ff6a6 深入了解
    '''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    return loss


# 计算识别精度
def get_accuracy(logits, labels):
    #
    acc = tf.nn.in_top_k(logits, labels, 1)
    acc = tf.cast(acc, tf.float32)
    acc = tf.reduce_mean(acc)
    return acc


# 训练
def training(loss, lr):
    train_ops = tf.train.RMSPropOptimizer(lr, 0.9).minimize(loss)
    return train_ops


def run_training():
    file_dir = 'local_data/'
    image, label = get_files(file_dir)
    image_batches, label_batches = get_batches(image, label, 28, 28, 10, 20)

    logits = cnn_model(image_batches)
    loss = get_loss(logits, label_batches)

    train_ops = training(loss, 0.001)
    acc = get_accuracy(logits, label_batches)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(2):
            print("run_training: step %d" % step)
            if coord.should_stop():
                break
            _, train_acc, train_loss = sess.run([train_ops, acc, loss])
            print("loss:{} , accuracy:{}".format(train_loss, train_acc))
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def get_files(file_dir):
    """
    :param filename: 文件夹名/
    :return: 包含文件名，标签名的列表
    """
    class_list = []
    label_list = []
    for train_class in os.listdir(file_dir):
        for pic in os.listdir(file_dir + train_class):
            class_list.append(file_dir + train_class+'/'+pic)
            label_list.append(train_class)
    temp = np.array([class_list, label_list])
    temp = temp.transpose()
    # shuffle the samples
    np.random.shuffle(temp)
    # after transpose, images is in dimension 0 and label in dimension 1
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    print(label_list)
    return image_list, label_list


def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
    """
    获取批次
    :param image: 图片列表
    :param label: 标签列表
    :param resize_w: 宽
    :param resize_h: 高
    :param batch_size: 批次大小
    :param capacity:
    :return: 图片，列表批次
    """
    # convert the list of images and labels to tensor
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    queue = tf.train.slice_input_producer([image, label])
    label = queue[1]
    # 读取图片
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c, channels=3)
    # resize
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
    # (x - mean) / adjusted_stddev 标准化像素深度
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    images_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])

    print(image_batch)
    return images_batch, labels_batch


run_training()
