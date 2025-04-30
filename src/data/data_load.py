import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
from botocore.exceptions import ClientError

BUCKET = "bucketest-1214"
PREFIX = "diffusion/"
CSV_FILE = "training.csv"

s3_client = boto3.client("s3")


def load_csv(file=CSV_FILE):
    """Loads the CSV file from S3"""
    print(f"📥 Loading CSV from S3: {file}")
    csv_obj = s3_client.get_object(Bucket=BUCKET, Key=PREFIX + file)
    return pd.read_csv(csv_obj['Body'], sep=";")


def send_csv(df, file_name):
    """Uploads the given DataFrame as a CSV to S3"""
    key = PREFIX + file_name
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    print(f"📤 CSV uploaded to S3: {file_name}")

def send_parquet(df, file_name):
    """Converts the DataFrame to Parquet and uploads it to S3"""
    key = PREFIX + file_name
    table = pa.Table.from_pandas(df)
    buffer = BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    print(f"📤 Parquet uploaded to S3: {file_name}")
