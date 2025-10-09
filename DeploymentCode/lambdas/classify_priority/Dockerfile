# Use the base image we created in Step 1
FROM 248612013410.dkr.ecr.ap-south-1.amazonaws.com/classifier-base:latest

COPY download_model.py .
RUN python3 download_model.py


COPY lambda_function.py .

CMD [ "lambda_function.handler" ]