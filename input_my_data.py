from PIL import Image
import numpy as np
import os

'''
created by Devin Zhang 2018/1/13
我的数据管理类
封装一些数据输入操作
'''

class MyDataManerger(object):

    filename = 'MY_data'

    img_list = np.array()
    label_list = np.array()

    def __init__(self, file_dir):
        self.file_dir = file_dir

    def next_batch_data(self, num):
        """
        :param num: 每批次的图片个数
        :return: 批次
        """
        for i in range(num):
            file_name = self.file_dir

            im = Image.open(file_name)

            # 预览图片
            print(im.show())

            # 剪切固定大小
            x_s = 28
            y_s = 28
            img = im.resize((x_s, y_s), Image.ANTIALIAS)

            # 转换为灰度图
            gray_img = img.convert('L')

            # 从tensor 对象转换为 python 数组
            im_arr = np.array(gray_img)

            # 打印矩阵
            print(im_arr)

            # 转换成一维向量
            nm = im_arr.reshape((1, 784))

            nm = nm.astype(np.float32)
            nm = np.multiply(nm, 1.0 / 255.0)

            return nm

    def pic_adapter(self):

        """'
        图片适配器
        """

