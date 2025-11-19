# Use the efficient, shared base image
FROM 248612013410.dkr.ecr.ap-south-1.amazonaws.com/classifier-base:latest

# We no longer need to copy requirements.txt or run pip install

# Copy and run the script to download THIS function's specific model
COPY download_model.py .
RUN python3 download_model.py

# Copy the handler code
COPY lambda_function.py .

# Set the command
CMD [ "lambda_function.lambda_handler" ]