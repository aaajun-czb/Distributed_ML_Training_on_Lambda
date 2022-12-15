## Introduction

Here's a Distributed ML training Application based on AWS Lambda.
It's based on Data Parallisam and S3 to achieve the data aggregation.

## Environment

AWS Lambda
DockerImage:Amazon Python:3.8
PyTorch:1.0+
Pycocotools

## Dataset

PennFudanPed

## Execution
```
# Environment Configuration: 
AWS account application, Lambda function creation, S3 creation, ECR creation.

# Upload Your Dataset to S3: 
Then, Simply modify the code structure according to the data set structure, and then enter the names of S3 in the Trigger Function event.

# Select the Model:
You can specify the network you want to train in get_model_instance_segmentation in pennfudanped.py, and pay attention to the corresponding network structure.

# Start the Train:
Test the Function blasting_trigger by sending the event.

```