import boto3
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import numpy as np
import utils
import urllib
import pickle

from pennfudanped import PennFudanDataset, get_transform, get_model_instance_segmentation
from engine import train_one_epoch, evaluate
from alexnet_cifar10 import AlexNet, ResNet, load_data

'''解压文件'''
def unzip_file(zip_name, dst_dir):
    # zip_name为带解压zip文件地址和名称, dst_dir为生成的文件夹
    fz = zipfile.ZipFile(zip_name, 'r')
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    fz.close()

'''聚合各Worker训练矩阵的函数'''
def scatter_reduce(s3_client, weights, epoch, worker_index, pattern_k, tmp_bucket, merged_bucket, output_file, batch = 0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    upload_start = time.time()
    vector = weights[0].reshape(1,-1)
    for i in range(1,len(weights)):
        vector = np.append(vector, weights[i].reshape(1,-1))
        # vector is supposed to be a 1-d numpy array
    num_all_values = vector.size
    num_values_per_agg = num_all_values // pattern_k
    residue = num_all_values

    # write partitioned vector to the shared memory, except the chunk charged by myself
    for i in range(pattern_k):
        if i != worker_index:
            offset = (num_values_per_agg * i) + min(residue, i)
            length = num_values_per_agg + (1 if i < residue else 0)
            # indicating the chunk number and which worker it comes from
            # format of key in tmp-bucket: chunkID_epoch_batch_workerID
            key = "{}_{}_{}_{}".format(i, epoch, batch, worker_index)
            s3_client.put_object(Bucket = tmp_bucket, Key = key, Body = vector[offset: offset + length].tobytes())
            logging.info(f"this is tmp {key} ")
    upload_time = time.time() - upload_start
    output_file.write(f"    Upload Time: {upload_time:.2f}\n")
    
    #每个worker都是aggregator
    merged_value = dict()
    aggre_start = time.time()
    my_offset = (num_values_per_agg * worker_index) + min(residue, worker_index)
    my_length = num_values_per_agg + (1 if worker_index < residue else 0)
    my_chunk = vector[my_offset: my_offset + my_length]
    # read and aggregate the corresponding chunk
    num_files = 0
    while num_files < pattern_k - 1:
        # 我猜测不同的worker训练时间相差一两分钟，有的等不到直接fail了，一直等待
        # if time.time()-time_stamp > 20:
        #     logging.info('Failed')
        #     return "failed"
        lists = s3_client.list_objects_v2(Bucket=tmp_bucket)
        if lists['KeyCount'] > 0:
            objects = lists['Contents']
        else:
            objects = None
        if objects is not None:
            for obj in objects:
                file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
                key_splits = file_key.split("_")
                # if it's the chunk I care and it is from the current step
                # format of key in tmp-bucket: chunkID_epoch_batch_workerID
                if key_splits[0] == str(worker_index) and key_splits[1] == str(epoch) and key_splits[2] == str(batch):
                    logging.info(f"get tmp {file_key} ")
                    data = s3_client.get_object(Bucket=tmp_bucket, Key = file_key)['Body'].read()
                    cur_data = np.frombuffer(data, dtype=vector.dtype)
                    my_chunk += cur_data
                    num_files += 1
                    s3_client.delete_object(Bucket = tmp_bucket, Key = file_key)
        time.sleep(1)
    # average weights
    my_chunk /= float(pattern_k)
    # write the aggregated chunk back
    # key format in merged_bucket: epoch_batch_chunkID
    key = "{}_{}_{}".format(epoch, batch, worker_index)
    s3_client.put_object(Bucket=merged_bucket, Key=key, Body = my_chunk.tobytes())
    logging.info(f"merged {key} ")
    merged_value[worker_index] = my_chunk
    aggre_time = time.time() - aggre_start
    output_file.write(f"    Aggre Time: {aggre_time:.2f}\n")

    # read other aggregated chunks
    download_start = time.time()
    num_merged_files = 0
    already_read_files = []
    if worker_index < pattern_k:
        total_files = pattern_k-1
    else:
        total_files = pattern_k
    while num_merged_files < total_files:
        lists = s3_client.list_objects_v2(Bucket=merged_bucket)
        if lists['KeyCount'] > 0:
            objects = lists['Contents']
        else:
            objects = None
        if objects is not None:
            for obj in objects:
                file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
                key_splits = file_key.split("_")
                # key format in merged_bucket: epoch_batch_chunkID
                # if not file_key.startswith(str(my_rank)) and file_key not in already_read:
                if key_splits[2] != str(worker_index) and key_splits[0] == str(epoch) and key_splits[1] == str(batch) and file_key not in already_read_files:
                    data = s3_client.get_object(Bucket=merged_bucket, Key = file_key)['Body'].read()
                    cur_data = np.frombuffer(data, dtype=vector.dtype)
                    merged_value[int(key_splits[2])] = cur_data
                    already_read_files.append(file_key)
                    num_merged_files += 1
                    logging.info(f"get merged {file_key} ")
        time.sleep(1)

    # reconstruct the whole vector
    new_vector = merged_value[0]
    for k in range(1, pattern_k):
        new_vector = np.concatenate((new_vector, merged_value[k]))
    result = dict()
    index = 0
    for k in range(len(weights)):
        lens = weights[k].size
        tmp_arr = new_vector[index:index + lens].reshape(weights[k].shape)
        result[k] = tmp_arr
        logging.info(f"type of result[k] {type(result[k])} ")
        index += lens
    download_time = time.time() - download_start
    output_file.write(f"    Download Time: {download_time:.2f}\n")
        
    return result

'''PennFudanPed数据集的Reduce的运行函数'''
def pennfudanped_reduce(train_net, pattern_k, worker_index, batch_size, epoches, l_r, memory, agg_mod, tmp_bucket, merged_bucket, train_bucket, output_bucket):
    # 连接s3,开启日志记录
    s3_client = boto3.client('s3')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 聚合模式,我只设了epoch，按batch数量的自行修改
    if agg_mod != 'epoch':
        num_batches = 'epoch'
    else:
        num_batches = 'epoch'

    # 输出本worker的工作信息到文件中
    output_file = open('/tmp/file.txt', 'w', encoding='utf-8')
    output_file.write(f"Worker Index: {worker_index}\n")
    output_file.write(f"Pattern K: {pattern_k}\n")
    output_file.write(f"Memory of Function: {memory}\n")
    output_file.write(f"Batch Size: {batch_size}\n")
    output_file.write(f"Learning Rate: {l_r}\n")
    output_file.write(f"Num of Epoches: {epoches}\n")
    output_file.write("Aggregate after each epoch.\n")

    # 从S3下载zip打包的数据集并解压
    prefix = "PennFudanPed"
    os_path = '/tmp/'+prefix
    s3_client.download_file(train_bucket, prefix+str(worker_index)+".zip", os_path+'.zip')
    unzip_file(os_path+'.zip', os_path)

    # 指定cuda,类数（Person和Background）,数据集传入定义好的Class中
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 2
    dataset = PennFudanDataset(os_path, get_transform(train=True))
    dataset_test = PennFudanDataset(os_path, get_transform(train=False))

    # 把数据集分割为train和test,原本数据集是170个,倒数50个作为test,我这里将50按pattern_k整除了一下
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-(50//pattern_k)])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-(50//pattern_k):])

    # 定义训练集和测试集
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=utils.collate_fn
    )

    # 得到模型
    model = get_model_instance_segmentation(num_classes)

    # construct an optimizer and a learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=l_r, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 这个是等所有function都start了才下一步的意思吗？
    # ready_file = open('/tmp/ready.txt','w',encoding='utf-8')
    # ready_file.close()
    # s3_client.upload_file('/tmp/ready.txt', train_bucket, 'ready' + str(worker_index))
    # flag = True
    # while flag:
    #     lists = s3_client.list_objects_v2(Bucket=train_bucket, Prefix = 'start')
    #     if lists['KeyCount'] == 1:
    #         flag = False
    #     else:
    #         time.sleep(1)
    
    # worker开始train
    t_start = time.time()
    epoch_end_file = open('/tmp/epoch_end_file.txt','w',encoding='utf-8') 
    epoch_end_file.close()
    logging.info(f"this is worker {worker_index} ")
    logging.info(f"batch size is {batch_size} ")
    for epoch in range(epoches): 
        logging.info(f"[{worker_index}]epoch is {epoch+1}")
        output_file.write(f"\n#Epoch {epoch+1}\n")
        train_start = time.time() # 记录Train时间
        # train for one epoch, printing every 10 iterations
        metric_logger=train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        output_file.write(f"\n#Train {metric_logger}\n")
        # update the learning rate
        lr_scheduler.step()
        # epoch的train完成，aggregate
        train_time = time.time() - train_start
        output_file.write(f'  Train Time of Epoch {epoch+1}: {train_time:.2f}\n')
        if agg_mod == 'epoch':
            weights = [param.data.numpy() for param in model.parameters()]
            merged_weights = scatter_reduce(s3_client, weights, epoch, worker_index, pattern_k, tmp_bucket, merged_bucket, output_file, batch_size)
            # 应该是把merged后的权重写入net中？
            for layer_index, param in enumerate(model.parameters()):
                param.data = torch.from_numpy(merged_weights[layer_index])
        output_file.write(f"  Current Total Time: {(time.time() - t_start):.2f}\n")
        #  这是在？每个epoch最后一个worker都结束并删除临时文件才会开始下一epoch
        s3_client.upload_file('/tmp/epoch_end_file.txt', train_bucket, 'epoch_end' + str(worker_index))
        end_epoch_lists = s3_client.list_objects_v2(Bucket=train_bucket, Prefix='epoch_end')
        if end_epoch_lists['KeyCount'] == pattern_k:   
            objects = end_epoch_lists['Contents']
            for obj in objects:
                s3_client.delete_object(Bucket=train_bucket, Key=obj['Key'])
            lists = s3_client.list_objects_v2(Bucket=merged_bucket, Prefix=f'{epoch+1}_')
            if lists['KeyCount'] > 0:
                    objects = lists['Contents']
            else:
                objects = None
            if objects is not None:
                for obj in objects:
                    s3_client.delete_object(Bucket=merged_bucket, Key=obj['Key'])
    training_time = time.time() - t_start
    output_file.write(f"\nTotal Time: {training_time:.2f}\n")
    logging.info(f'[{worker_index}]Finished Training')

    # 重新用模型进行evaluate看看
    result_info = evaluate(model, data_loader_test, device=device)
    output_file.write(f"\n#Evaluator {result_info}\n")
    output_file.close()
    logging.info(f'[{worker_index}]Finished Testing ')
    path = 'W' + str(worker_index) + '_B' + str(batch_size) + '_E' + str(epoches) + '_M' + str(memory) + '_K' + str(pattern_k) + '_A' + str(num_batches) + '_' + str(train_net) + '.txt'
    s3_client.upload_file('/tmp/file.txt', output_bucket, path)
    PATH = '/tmp/cifar_net.pth'
    torch.save(model.state_dict(), PATH) 

    return {"result": "succeed!"}

'''cifar10数据集的Reduce的运行函数'''
def cifar10_reduce(train_net, pattern_k, worker_index, batch_size, epoches, l_r, memory, agg_mod, tmp_bucket, merged_bucket, train_bucket, output_bucket):
    # 连接s3
    s3_client = boto3.client('s3')
    # 开启日志记录
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 聚合模式,我只设了epoch，按batch数量的自己改改
    if agg_mod != 'epoch':
        # splits = agg_mod.split("_")
        # num_batches = int(splits[1])
        num_batches = 'epoch'
    else:
        num_batches = 'epoch'

    # 输出本worker的工作信息到文件中
    output_file = open('/tmp/file.txt', 'w', encoding='utf-8')
    output_file.write(f"Worker Index: {worker_index}\n")
    output_file.write(f"Pattern K: {pattern_k}\n")
    output_file.write(f"Memory of Function: {memory}\n")
    output_file.write(f"Batch Size: {batch_size}\n")
    output_file.write(f"Learning Rate: {l_r}\n")
    output_file.write(f"Num of Epoches: {epoches}\n")
    output_file.write(f"Num of Batches: {num_mini_batches}\n")
    output_file.write(f"Num of Samples: {num_pics}\n")
    output_file.write("Aggregate after each epoch.\n")
    # if agg_mod == 'epoch':
    #     output_file.write("Aggregate after each epoch.\n")
    # else:
    #     output_file.write(f"Aggregate every {num_batches} batches.\n")

    # 从S3下载zip打包的数据集并解压
    prefix = "cifar-10-pictures-"
    s3_client.download_file(train_bucket, prefix+str(worker_index)+".zip", '/tmp/cifar-10-pictures.zip')
    s3_client.download_file(train_bucket, prefix+"test.zip", '/tmp/cifar-10-pictures-test.zip')
    unzip_file('/tmp/cifar-10-pictures.zip', '/tmp/cifar-10-pictures')
    unzip_file('/tmp/cifar-10-pictures-test.zip', '/tmp/cifar-10-pictures-test')

    # 指定网络，读取数据集，指定torch的一些参数
    if train_net == 'alex':
        net = AlexNet()
    else:
        net = ResNet()
    trainloader, testloader = load_data(batch_size, '/tmp/cifar-10-pictures', '/tmp/cifar-10-pictures-test')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=l_r, momentum=0.9)

    # 每个worker的图片数量，以及batch数
    num_pics = 50000 // pattern_k
    num_mini_batches = num_pics // batch_size
    # count_batches = 0 用于按batch数量aggregate的

    # 这个是等所有function都start了才下一步的意思吗？
    ready_file = open('/tmp/ready.txt','w',encoding='utf-8')
    ready_file.close()
    s3_client.upload_file('/tmp/ready.txt', train_bucket, 'ready' + str(worker_index))
    flag = True
    while flag:
        lists = s3_client.list_objects_v2(Bucket=train_bucket, Prefix = 'start')
        if lists['KeyCount'] == 1:
            flag = False
        else:
            time.sleep(1)
    
    # worker开始train
    t_start = time.time()
    epoch_end_file = open('/tmp/epoch_end_file.txt','w',encoding='utf-8') 
    epoch_end_file.close()
    batches_time = time.time()
    logging.info(f"this is worker {worker_index} ")
    logging.info(f"batch size is {batch_size} ")
    for epoch in range(epoches): 
        logging.info(f"[{worker_index}]epoch is {epoch+1}")
        output_flag = True # 用作每一个epoch结尾将训练结果输出到txt中
        output_file.write(f"\n#Epoch {epoch+1}\n")
        train_start = time.time() # 记录Train时间
        running_loss = 0.0 # 记录本次epoch的loss
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data      # get the inputs; data is a list of [inputs, labels]
            optimizer.zero_grad()      # zero the parameter gradients
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 我默认按epoch aggregate了，把下面按batch数量注释掉了
            # count_batches += 1
            # if agg_mod != 'epoch' and count_batches % num_batches == 0:
            #     # scatter_reduce every 'num_batches' batches
            #     weights = [param.data.numpy() for param in net.parameters()]
            #     merged_weights = scatter_reduce(
            #         weights, epoch+1, worker_index, pattern_k, i+1)
            #     for layer_index, param in enumerate(net.parameters()):
            #         param.data = torch.from_numpy(merged_weights[layer_index])
        # epoch的train完成，aggregate
        train_time = time.time() - train_start
        output_file.write(f'  Train Time of Epoch {epoch+1}: {train_time:.2f}\n')
        if agg_mod == 'epoch':
            weights = [param.data.numpy() for param in net.parameters()]
            merged_weights = scatter_reduce(s3_client, weights, epoch, worker_index, pattern_k, tmp_bucket, merged_bucket, output_file)
            # 应该是把merged后的权重写入net中？我不懂这几个网络结构，没改这些
            for layer_index, param in enumerate(net.parameters()):
                param.data = torch.from_numpy(merged_weights[layer_index])
        output_file.write(f"  Loss of Epoch {epoch+1}: {running_loss/(i+1)}\n")
        output_file.write(f"  Current Total Time: {(time.time() - t_start):.2f}\n")
        #  这是在？每个epoch所有worker都结束并删除临时文件才会开始下一epoch？有任何必要性吗？
        s3_client.upload_file('/tmp/epoch_end_file.txt', train_bucket, 'epoch_end' + str(worker_index))
        end_epoch_lists = s3_client.list_objects_v2(Bucket=train_bucket, Prefix='epoch_end')
        if end_epoch_lists['KeyCount'] == pattern_k:   
            objects = end_epoch_lists['Contents']
            for obj in objects:
                s3_client.delete_object(Bucket=train_bucket, Key=obj['Key'])
            lists = s3_client.list_objects_v2(Bucket=merged_bucket, Prefix=f'{epoch+1}_')
            if lists['KeyCount'] > 0:
                    objects = lists['Contents']
            else:
                objects = None
            if objects is not None:
                for obj in objects:
                    s3_client.delete_object(Bucket=merged_bucket, Key=obj['Key'])
    training_time = time.time() - t_start
    output_file.write(f"\nTotal Time: {training_time:.2f}\n")
    logging.info(f'[{worker_index}]Finished Training')

    # 我觉得test由一个worker做就行了，不需要所有worker重复test
    if worker_index == pattern_k-1:
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
        output_file.write(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
        output_file.close()
        logging.info(f'[{worker_index}]Finished Testing ')
        path = 'N' + str(pattern_k) + '_B' + str(batch_size) + '_E' + str(epoches) + '_M' + str(memory) + '_K' + str(pattern_k) + '_A' + str(num_batches) + '_' + str(train_net) + '.txt'
        s3_client.upload_file('/tmp/file.txt', output_bucket, path)
        PATH = '/tmp/cifar_net.pth'
        torch.save(net.state_dict(), PATH) 

    return {"result": "succeed!"}