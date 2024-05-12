import boto3
import botocore
from botocore.exceptions import ClientError
import zipfile

def download_data():
    """Download training data from Amazon S3 bucket
       Used when run as a ECS task
    """

    s3 = boto3.client('s3')
    bucket_name = 'cifar10-mlops-bucket'
    file_key = 'data.zip'
    local_file_path = 'data.zip'
    extract_to = './'

    botocore.session.Session().set_debug_logger()

    # Download the file from S3
    try:
        s3.download_file(bucket_name, file_key, local_file_path)
        print("File downloaded successfully.")

        # Extract the contents of the zip file
        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Zip file extracted successfully to '{extract_to}'.")

    except ClientError as e:
        print(f"An error occurred while downloading training data {e}")

def upload_model():
    """Upload pre-trained model to S3 bucket
    """

    s3_client = boto3.client("s3")
    file_path = "model.pth"
    bucket_name = "cifar10-mlops-bucket"
    object_key = "model.pth"

    try:
        s3_client.upload_file(file_path, bucket_name, object_key)
        print(f"Uploaded {file_path} to {bucket_name}/{object_key}")
    except ClientError as e:
        print(f"An error occurred while uploading {e}")
