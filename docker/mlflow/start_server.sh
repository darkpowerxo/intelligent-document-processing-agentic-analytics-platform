#!/bin/bash

# MLflow server startup script

echo "Starting MLflow server..."

# Install MLflow and dependencies
pip install mlflow==2.8.1 boto3 psycopg2-binary

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
python << EOF
import psycopg2
import time
import sys

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        conn = psycopg2.connect(
            host='postgres',
            port=5432,
            database='ai_demo',
            user='ai_demo',
            password='ai_demo_password'
        )
        conn.close()
        print("PostgreSQL is ready!")
        sys.exit(0)
    except psycopg2.OperationalError:
        retry_count += 1
        time.sleep(2)

print("Failed to connect to PostgreSQL after 60 seconds")
sys.exit(1)
EOF

# Wait for MinIO to be ready
echo "Waiting for MinIO..."
python << EOF
import boto3
import time
import sys
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        # Try to list buckets to test connection
        s3_client.list_buckets()
        print("MinIO is ready!")
        sys.exit(0)
    except (ClientError, NoCredentialsError, EndpointConnectionError):
        retry_count += 1
        time.sleep(2)

print("Failed to connect to MinIO after 60 seconds")
sys.exit(1)
EOF

# Create MLflow artifacts bucket in MinIO
python << EOF
import boto3
from botocore.exceptions import ClientError

try:
    s3_client = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin'
    )
    
    # Create bucket if it doesn't exist
    try:
        s3_client.create_bucket(Bucket='mlflow-artifacts')
        print("Created MLflow artifacts bucket")
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyExists':
            print("MLflow artifacts bucket already exists")
        else:
            raise
            
except Exception as e:
    print(f"Error setting up MinIO bucket: {e}")
EOF

# Start MLflow server
echo "Starting MLflow tracking server..."
mlflow server \
    --backend-store-uri postgresql://ai_demo:ai_demo_password@postgres:5432/ai_demo \
    --default-artifact-root s3://mlflow-artifacts/ \
    --artifacts-destination s3://mlflow-artifacts/ \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000
