import os
import shutil

from reduce import pennfudanped_reduce

def handler(event, context):
    # training setting
    train_net = event['train_net']
    pattern_k = int(event['pattern_k'])             # 我这里worker就是aggregator了，所以其实就是worker数量
    worker_index = int(event['worker_index'])       # index of worker
    batch_size = int(event['batch_size'])
    epoches = int(event['epoches'])
    l_r = float(event['l_r'])
    memory = int(event['memory'])
    agg_mod = event['agg_mod']
    tmp_bucket = event['tmp_bucket']
    merged_bucket = event['merged_bucket']
    train_bucket = event['train_bucket']
    output_bucket = event['output_bucket']

    # 删除文件夹，我不知道是什么意思，lambda创建函数难道不会清空tmp吗
    if os.path.exists('/tmp/PennFudanPed'):
        shutil.rmtree('/tmp/PennFudanPed')

    # 启动reduce
    result = pennfudanped_reduce(train_net, pattern_k, worker_index, batch_size, epoches, l_r, memory, agg_mod, tmp_bucket, merged_bucket, train_bucket, output_bucket)

    return result