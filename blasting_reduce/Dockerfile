# FROM base_image

# RUN yum -y install gcc
# RUN pip3 install pycocotools
FROM blasting_reduce:v1

COPY hub.py /var/lang/lib/python3.8/site-packages/torch/hub.py

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY reduce.py ${LAMBDA_TASK_ROOT}
COPY alexnet_cifar10.py ${LAMBDA_TASK_ROOT}
COPY pennfudanped.py ${LAMBDA_TASK_ROOT}
COPY transforms.py ${LAMBDA_TASK_ROOT}
COPY engine.py ${LAMBDA_TASK_ROOT}
COPY coco_eval.py ${LAMBDA_TASK_ROOT}
COPY coco_utils.py ${LAMBDA_TASK_ROOT}
COPY utils.py ${LAMBDA_TASK_ROOT}
# COPY resnet50_fpn.pth ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]