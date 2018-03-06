# -*- coding: utf-8 -*-

import os
from scipy.cluster.vq import *
from scipy.misc import imresize

from pylab import *

from PIL import Image

def calculate_square(file, k, steps):

    im = array(Image.open(file))
    '''
    计算每块的大小
    '''
    dx = im.shape[0] / steps
    dy = im.shape[1] / steps

    '''
    计算每个区域的颜色特征
    '''
    features = []
    for x in range(steps):
        for y in range(steps):
            R = mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 0])
            G = mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 1])
            B = mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 2])
            features.append([R, G, B])
    features = array(features, 'f')
    '''
    聚类， k是聚类数目
    '''
    cent, var = kmeans(features, k)
    code, distance = vq(features, cent)

    '''
    用聚类标记创建图像
    '''
    codeim = code.reshape(steps, steps)
    codeim = imresize(codeim, im.shape[:2], 'nearest')
    return codeim


def handle(dir ,filename):
    path = dir + filename

    # 参数
    k = 15
    steps = 128

    # 方形块对图片的像素进行聚类
    codeim = calculate_square(path, k, steps)

    imsave(dir + 'save_dir' + os.sep + filename.split('.')[0] + '_handled.' + filename.split('.')[1] , codeim)

#这个函数把所有当前目录下的图片处理
def readAndSave(dir):

    files = os.listdir(dir)

    if not os.path.exists(dir+'save_dir/'):
        os.mkdir(dir+'save_dir/')

    for i in files:
        if os.path.isfile(dir+i):
            handle(dir,i)

#这个函数可以把处理的图像show出来
def Run():
    #图像文件 路径
    dir = './data/'
    file =  '2.jpg'
    path = dir + file

    im = array(Image.open(path))

    #参数
    k = 15
    steps = 128

    #显示原图empire.jpg
    figure()
    subplot(121)
    title('source')
    imshow(im)

    #方形块对图片的像素进行聚类
    codeim= calculate_square(path, k, steps)
    imsave(dir+'handled_'+file,codeim)
    subplot(122)
    title('steps = 200 K = '+str(k));
    imshow(codeim)

    show()


#输入图像所在目录即可
readAndSave('./data/')