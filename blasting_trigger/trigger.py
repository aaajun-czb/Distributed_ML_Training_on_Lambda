import os
import pickle
import numpy as np 
import zipfile
import shutil

from operation_s3 import download_s3_folder

'''打开数据集文件'''
def unpickle(path):
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

'''压缩某个地址的文件夹'''
def zip_file(src_dir, zip_name):
    z = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        fpath = dirpath.replace(src_dir, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath + filename)
    z.close()

'''下载某个路径的pennfudanped的数据集并将给划分给worker数量'''
def pennfudanped_partition(os_path, num_pics_one, pattern_k):
    # 导出文件夹中文件的排序list
    imgs = list(sorted(os.listdir(os.path.join(os_path, "PNGImages"))))
    masks = list(sorted(os.listdir(os.path.join(os_path, "PedMasks"))))
    txts = list(sorted(os.listdir(os.path.join(os_path, "Annotation"))))
    for i, img in enumerate(imgs):
        # 前num_pics_one张为0号worker,依次递推,多的全给最后一个worker
        j = i // num_pics_one
        if j == pattern_k:
            shutil.copy(os.path.join(os_path, "PNGImages", img), os.path.join(os_path+str(j-1), "PNGImages", img))
        else:
            shutil.copy(os.path.join(os_path, "PNGImages", img), os.path.join(os_path+str(j), "PNGImages", img))
    for i, mask in enumerate(masks):
        j = i // num_pics_one
        if j == pattern_k:
            shutil.copy(os.path.join(os_path, "PedMasks", mask), os.path.join(os_path+str(j-1), "PedMasks", mask))
        else:
            shutil.copy(os.path.join(os_path, "PedMasks", mask), os.path.join(os_path+str(j), "PedMasks", mask))
    for i, txt in enumerate(txts):
        j = i // num_pics_one
        if j == pattern_k:
            shutil.copy(os.path.join(os_path, "Annotation", txt), os.path.join(os_path+str(j-1), "Annotation", txt))
        else:
            shutil.copy(os.path.join(os_path, "Annotation", txt), os.path.join(os_path+str(j), "Annotation", txt))

'''下载某个路径的cifar10的train数据集并将给划分给worker数量'''
def cifar10_partition_train(path, source_bucket, prefix, s3_client, label_name, order, num_pics_one):
    # 用法 s3.download_file('BUCKET_NAME', 'OBJECT_NAME', 'FILE_NAME')
    '''需要更换为我们爆破数据的数据集'''
    s3_client.download_file(source_bucket, path, "/tmp/"+path)
    data_batch = unpickle("/tmp/"+path)
    # 读取数据的label和data
    cifar_label = data_batch[b'labels']
    cifar_data = data_batch[b'data']
    # 把字典的值转成array格式，方便操作
    cifar_label = np.array(cifar_label)
    cifar_data = np.array(cifar_data)

    # cifar每个batch 10000张图片，根据数据集需要修改下面range，以及整理数据的方法
    '''这里需要根据爆破数据集的实际情况改我们要怎么存储划分后的数据'''
    for i in range(10000):
        image = cifar_data[i]
        image = image.reshape(-1, 1024)
        r = image[0, :].reshape(32, 32)  # 红色分量
        g = image[1, :].reshape(32, 32)  # 绿色分量
        b = image[2, :].reshape(32, 32)  # 蓝色分量
        img = np.zeros((32, 32, 3))
        #RGB还原成彩色图像
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        # 整除，前num_pics_one张图为0，然后依次递推，最后的图像全分给最后一个
        j = order[cifar_label[i]] // num_pics_one
        cv2.imwrite("/tmp/"+ prefix + str(j) + "/" + str(label_name[cifar_label[i]]) + "/" + str(label_name[cifar_label[i]]) + "_" + str(order[cifar_label[i]]) + ".jpg", img)
        # 标注这个class的第几张图像了
        order[cifar_label[i]] += 1

'''下载某个路径的test数据集并储存'''
def partition_test(path, source_bucket, prefix, s3_client, label_name):
    '''需要更换为我们爆破数据的数据集'''
    s3_client.download_file(source_bucket, path, "/tmp/"+path)
    data_batch = unpickle("/tmp/"+path)
    cifar_label = data_batch[b'labels']
    cifar_data = data_batch[b'data']
    cifar_label = np.array(cifar_label)
    cifar_data = np.array(cifar_data)

    for i in range(10000):
        image = cifar_data[i]
        image = image.reshape(-1, 1024)
        r = image[0, :].reshape(32, 32)  # 红色分量
        g = image[1, :].reshape(32, 32)  # 绿色分量
        b = image[2, :].reshape(32, 32)  # 蓝色分量
        img = np.zeros((32, 32, 3))
        #RGB还原成彩色图像
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        cv2.imwrite("/tmp/"+ prefix + "/" + str(label_name[cifar_label[i]]) + "/" + str(label_name[cifar_label[i]]) + "_" + str(i) + ".jpg", img)

def pennfudanped_trigger(pattern_k, s3_client, s3_resource, source_bucket, train_bucket):
    # 根据worker数量创建临时存放文件夹
    prefix = 'PennFudanPed'
    os_path = '/tmp/' + prefix
    if not os.path.exists(os_path):
        os.mkdir(os_path)
    for i in range(pattern_k):
        if not os.path.exists(os_path + str(i)):
            os.mkdir(os_path + str(i))
            os.mkdir(os_path + str(i) + '/PNGImages')
            os.mkdir(os_path + str(i) + '/PedMasks')
            os.mkdir(os_path + str(i) + '/Annotation')

    '''通过mod取余来分配会不会更好？这样每个worker最多多一张？'''
    # fudan74张图片，penn96张图片，共170张图片
    num_pics_one = 170 // pattern_k
    # 从s3里下载数据集然后划分
    download_s3_folder(s3_resource, source_bucket, prefix, os_path)
    pennfudanped_partition(os_path, num_pics_one, pattern_k)
    # 压缩文件夹然后上传
    for i in range(pattern_k):
        zip_file(os_path + str(i), os_path + str(i)+'.zip')
        s3_client.upload_file(os_path + str(i)+'.zip', train_bucket, prefix+str(i)+'.zip')

def cifar10_trigger(num_workers, batch_size, epoches, l_r, s3_client, source_bucket, train_bucket):
    '''之后根据爆破标签替换'''
    label_name = ['airplane', 'automobile', 'brid', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 根据worker数量创建按标签分类的文件夹
    prefix = 'cifar-10-pictures-'
    os_path = '/tmp/' + prefix
    for i in range(num_workers):
        if not os.path.exists(os_path + str(i)):
            os.mkdir(os_path + str(i))
            for label in label_name:
                os.mkdir(os_path +str(i) + '/' + label)
    os_path = '/tmp/' + prefix + 'test'
    if not os.path.exists(os_path + 'test'):
        os.mkdir(os_path)
        for label in label_name:
            os.mkdir(os_path + label)

    '''通过mod取余来分配会不会更好？这样每个worker最多多一张？'''
    order = [0]*10 # 创造了一个1行10列的0矩阵，用以储存每个类分到了第几张，从而可以计算该给第几个worker了
    num_pics_one = 5000 // num_workers # 10个class，每个5000张，平分给每个worker，除了最后一个worker可能会多分几张因为不能整除
    # 从s3里下载train数据集，cifar10数据集从data_batch_1到5
    for batch in range(1, 6):
        cifar10_partition_train("data_batch_"+str(batch), source_bucket, prefix, s3_client, label_name, order, num_pics_one)
    # 压缩文件夹然后上传
    for i in range(num_workers):
        zip_file("/tmp/"+prefix+str(i), "/tmp/"+prefix+str(i)+'.zip')
        # 用法为s3.upload_file(sourcefile, bucket, key, callback, extra_args)
        s3_client.upload_file("/tmp/"+prefix+str(i)+'.zip', train_bucket, prefix+str(i)+'.zip')

    # 从s3里下载test数据集
    prefix = 'cifar-10-pictures-test'
    partition_test("test_batch", source_bucket, prefix, s3_client, label_name)
    zip_file("/tmp/"+prefix, "/tmp/"+prefix+'.zip')
    s3_client.upload_file("/tmp/"+prefix+'.zip', train_bucket, prefix+'.zip')