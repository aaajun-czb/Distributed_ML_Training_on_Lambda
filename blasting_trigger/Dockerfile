FROM base_image

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY trigger.py ${LAMBDA_TASK_ROOT}
COPY operation_s3.py ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]