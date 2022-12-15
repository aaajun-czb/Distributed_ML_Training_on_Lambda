import urllib
import boto3
import os

def clear_bucket(s3_client, bucket_name):
    lists = s3_client.list_objects_v2(Bucket=bucket_name)
    if lists['KeyCount'] > 0:
        objects = lists['Contents']
    else:
        objects = None
    if objects is not None:
        obj_names = []
        for obj in objects:
            file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
            obj_names.append(file_key)
        if len(obj_names) >= 1:
            obj_list = [{'Key': obj} for obj in obj_names]
            s3_client.delete_objects(Bucket=bucket_name, Delete={'Objects': obj_list})
    return True

def download_s3_folder(s3, bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        s3: s3_resource
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)