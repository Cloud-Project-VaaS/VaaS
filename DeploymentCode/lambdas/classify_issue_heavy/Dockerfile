# Start from the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.12

# Copy the requirements file
COPY requirements.txt .

# Install the specified packages
RUN pip install -r requirements.txt

# Copy and run the download script to bake the model into the image
# This happens only ONCE during the build process
COPY download_model.py .
RUN python3 download_model.py

# Copy the Lambda handler code
COPY lambda_function.py .

# Set the command to run your handler function
CMD [ "lambda_function.lambda_handler" ]