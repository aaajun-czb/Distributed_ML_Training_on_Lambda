import logging
import boto3
import json
import time

from trigger import pennfudanped_trigger
from operation_s3 import clear_bucket

def handler(event, context):
    # 传入event参数
    pattern_k = int(event['pattern_k'])             # 划分的份数，可以认为是worker的数量，这里不搞aggregator了
    batch_size = int(event['batch_size'])           # batch size，建议为128
    epoches = int(event['epoches'])                 # epoch的次数
    l_r = float(event['l_r'])                       # Learning rate，建议为0.01到0.005
    agg_mod = event['agg_mod']                      # aggregation的模式，可选值为'epoch'
    memory = event['memory']                        # 指定的Reduce Function的内存
    train_net = event['train_net']                  # 训练网络，可选值为'r-cnn'
    source_bucket = event['source_bucket']          # 存储原始数据集的s3桶名称
    tmp_bucket = event['tmp_bucket']                # 存储中间数据的s3桶名称
    merged_bucket = event['merged_bucket']          # 存储融合数据的s3桶名称
    train_bucket = event['train_bucket']            # 存储训练数据的s3桶名称
    output_bucket = event['output_bucket']          # 存储结果的s3桶名称
    
    # 开启日志记录
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 连接s3并清空
    s3_client = boto3.client('s3')
    clear_bucket(s3_client, tmp_bucket)
    clear_bucket(s3_client, merged_bucket)
    clear_bucket(s3_client, train_bucket)
    clear_bucket(s3_client, output_bucket)

    # 把数据集划分
    s3_resource = boto3.resource('s3')
    pennfudanped_trigger(pattern_k, s3_client, s3_resource, source_bucket, train_bucket)

    # Invoke Lambda的Reduce的Function
    payload = dict()
    payload['train_net'] = train_net
    payload['pattern_k'] = pattern_k
    payload['batch_size'] = batch_size
    payload['epoches'] = epoches
    payload['l_r'] = l_r
    payload['agg_mod'] = agg_mod
    payload['memory'] = memory
    payload['tmp_bucket'] = tmp_bucket
    payload['merged_bucket'] = merged_bucket
    payload['train_bucket'] = train_bucket
    payload['output_bucket'] = output_bucket
    lambda_client = boto3.client('lambda')
    for i in range(pattern_k):
        payload['worker_index'] = i
        lambda_client.invoke(FunctionName='blasting_reduce',
                             InvocationType='Event',
                             Payload=json.dumps(payload))
        logging.info(f"function {i} has been invoked")

    # 所有worker的函数都启动后，写个start的文件做标志，通知所有worker开始同步train了
    # file = open('/tmp/start.txt','w',encoding='utf-8')
    # file.close()
    # flag = True
    # while flag:
    #     lists = s3_client.list_objects_v2(Bucket=train_bucket, Prefix = 'ready')
    #     if lists['KeyCount'] == pattern_k:
    #         flag = False
    #     else:
    #         time.sleep(1)
    # clear_bucket(s3_client, train_bucket)
    # s3_client.upload_file('/tmp/start.txt', train_bucket, 'start')