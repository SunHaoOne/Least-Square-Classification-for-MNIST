# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 07:41:38 2020

@author: 孙昊一
"""

import numpy as np
import struct
import prettytable as pt
import time
# 训练集文件
train_images_idx3_ubyte_file = r'D:\2pic\train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = r'D:\2pic\train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = r'D:\2pic\t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = r'D:\2pic\t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def pretreat(train_labels,test_labels,train_images,test_images):
    
    train_images_column=train_images.reshape(60000,784,1)
    test_images_column=test_images.reshape(10000,784,1)
    train_labels=train_labels.reshape(60000,1)
    test_labels=test_labels.reshape(10000,1)
    
    for i in range(len(train_labels)):
        if train_labels[i] == 0:
            train_labels[i] =1
        elif train_labels[i] != 0:
            train_labels[i] = -1 ## 5923个0 /60000 约1/10 正确
        
    for i in range(len(test_labels)):
        if test_labels[i] == 0:
            test_labels[i] =1
        elif test_labels[i] != 0:
            test_labels[i] = -1 ## 980个0 /10000 约1/10  正确
            
    train_images_2D=train_images_column.reshape(60000,784)
    test_images_2D=test_images_column.reshape(10000,784)
    train_images_2DT=train_images_2D.T
    test_images_2DT=test_images_2D.T   
    
    return train_labels,test_labels,train_images_2DT,test_images_2DT

def show_result(labels,result,dataset):
    TP=0 #正类预测为正类
    FN=0 #正类预测为负类   
    FP=0 #负类预测为正类
    TN=0 #负类预测为负类
    for i in range(len(labels)):
        if labels[i]==1 and result[i]==1:
            TP=TP+1
        elif labels[i]==1 and result[i]==-1:
            FN=FN+1
        elif labels[i]==-1 and result[i]==1:
            FP=FP+1
        elif labels[i]==-1 and result[i]==-1:
            TN=TN+1   
    tb = pt.PrettyTable()
    tb.field_names = [dataset,"Predicted y=1","Prediected y=-1","Total"]
    tb.add_row(["y=+1",TP,FN,TP+FN])
    tb.add_row(["y=-1",FP,TN,FP+TN])
    tb.add_row(["All",TP+FP,FN+TN,TP+FP+FN+TN])
    print(tb)   
    error = (FN+FP)/(TP+FP+FN+TN) * 100
    print('错误率为',"%.3f" % error,'%')
    print('\n')
    
def tran1or0(result): ##这个函数来把大于0的转换为1，小于0的转换为0
    result[result>0]=1
    result[result<0]=-1
    result.reshape(len(result),1)    
    return result

if __name__ == '__main__':
    start1 = time.time()
    
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()  
    [train_labels,test_labels,train_images_2DT,test_images_2DT]=pretreat(train_labels,test_labels,train_images,test_images)
    tt=0
    index=[]
    train_image_feature=np.zeros([493,60000])
    for i in range (784):
        non_zero = np.linalg.norm(train_images_2DT[i,:], ord=0) 
        if non_zero >= 600:
            train_image_feature[tt,:]=train_images_2DT[i,:]
            tt=tt+1
            index.append(i)
    test_image_feature=np.zeros([493,10000])
    test_image_feature=test_images_2DT[index,:]
    A=np.hstack([np.ones([60000,1]),train_image_feature.T])
    A_test=np.hstack([np.ones([10000,1]),test_image_feature.T])
    b=train_labels
    b_test=test_labels
    print('进行QR分解中...')
    q,r = np.linalg.qr(A)
    print('已完成QR分解')
    print('\n')
    x=np.linalg.pinv(r).dot(q.T.dot(b))
    result=A.dot(x)
    tran1or0(result)
    show_result(train_labels,result,'Train_dataset')
    result_test=A_test.dot(x)
    tran1or0(result_test)
    show_result(test_labels,result_test,'Test_dataset')##
    end1 = time.time()
    print("运间:%.2f秒" % (end1 - start1))
    
    start2 = time.time()
    Random_feature=np.random.choice([-1,1],(5000,494))
    A_random=A.dot(Random_feature.T)
    A_random[A_random<0]=0 
    A_add=np.hstack([np.ones([60000,1]),train_image_feature.T,A_random])
    print('进行QR分解中...')
    q_add,r_add = np.linalg.qr(A_add)
    print('已完成QR分解')
    print('对x进行求解中...')
    x_add=np.linalg.pinv(r_add).dot(q_add.T.dot(b))
    print('已求得x的解')
    result_add=A_add.dot(x_add)
    print('已完成结果预测')
    tran1or0(result_add)       
    show_result(train_labels,result_add,'Train_dataset_ADD')  
    A_random_test=np.dot(A_test,Random_feature.T)##10000*50000
    A_random_test[A_random_test<0]=0 
    A_add_test=np.hstack([np.ones([10000,1]),test_image_feature.T,A_random_test]) ## 10000*5494
    result_add_test=A_add_test.dot(x_add)
    tran1or0(result_add_test)
    show_result(test_labels,result_add_test,'Test_dataset_ADD')
    
    
    end2 = time.time()
    print("运行时间:%.2f秒" % (end2 - start2))
 

       
    
