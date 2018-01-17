"""
主函数
"""
import numpy as np
import tensorflow as tf
import input_local_data as ild
import training_graph as tg

session = tf.InteractiveSession()

input_data = ild.InputLocalData('local_data/')
img_batch, lab_batch = input_data.get_batches(resize_w=28, resize_h=28,
                                              batch_size=5, capacity=20)

graph = tg.TrainingGraph(channels=3, keep_prob=1)
train_step, acc = graph.build_graph()

# init all variables
init = tf.global_variables_initializer()
session.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=session, coord=coord)
try:
    for step in np.arange(2):
        print("run_training: step %d" % step)
        if coord.should_stop():
            break
        train_step.run(feed_dict={graph.img_local_h: img_batch, graph.lab_local_h: lab_batch})
        print("accuracy:{}".format(acc.eval(feed_dict={graph.img_local_h: img_batch, graph.lab_local_h: lab_batch})))
except tf.errors.OutOfRangeError:
    print("Done!!!")
finally:
    coord.request_stop()
coord.join(threads)