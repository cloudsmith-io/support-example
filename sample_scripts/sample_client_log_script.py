import boto3
import gzip
import io
import sys
import json
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
logger = glueContext.get_logger()

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'source_bucket', 'source_log_file', 'target_bucket'])

job.init(args['JOB_NAME'], args)

# Define schema for input nested JSON data
schema = StructType([
    StructField("bytes", IntegerType(), True),
    StructField("datetime", StringType(), True),
    StructField("edge", StringType(), True),
    StructField("eula", StructType([
        StructField("identifier", StringType(), True),
        StructField("number", StringType(), True)
    ]), True),
    StructField("format", StringType(), True),
    StructField("host", StringType(), True),
    StructField("ip_address", StringType(), True),
    StructField("location", StructType([
        StructField("city", StringType(), True),
        StructField("continent", StringType(), True),
        StructField("country", StringType(), True),
        StructField("country_code", StringType(), True),
        StructField("latitude", StringType(), True),
        StructField("longitude", StringType(), True)
    ]), True),
    StructField("method", StringType(), True),
    StructField("namespace", StringType(), True),
    StructField("package", StructType([
        StructField("identifier", StringType(), True),
        StructField("name", StringType(), True),
        StructField("tags", StructType([
            StructField("version", StringType(), True)
        ]), True),
        StructField("version", StringType(), True)
    ]), True),
    StructField("recorded", StringType(), True),
    StructField("referer", StringType(), True),
    StructField("repository", StringType(), True),
    StructField("request_id", StringType(), True),
    StructField("status", IntegerType(), True),
    StructField("token", StructType([
        StructField("identifier", StringType(), True),
        StructField("name", StringType(), True),
        StructField("metadata", StringType(), True)
    ]), True),
    StructField("uri", StringType(), True),
    StructField("user", StructType([
        StructField("identifier", StringType(), True),
        StructField("username", StringType(), True)
    ]), True),
    StructField("user_agent", StructType([
        StructField("browser", StringType(), True),
        StructField("device", StringType(), True),
        StructField("os", StringType(), True),
        StructField("raw", StringType(), True)
    ]), True)
])


def process_gzip_file(bucket, key):
    s3 = boto3.client('s3')
    zip_obj = s3.get_object(Bucket=bucket, Key=key)
    buffer = io.BytesIO(zip_obj["Body"].read())

    with gzip.GzipFile(fileobj=buffer, mode='rb') as gz_file:
        json_content = gz_file.read().decode('utf-8')

    json_data = [json.loads(line) for line in json_content.splitlines()]
    return spark.createDataFrame(json_data, schema=schema)


# Extract nested fields with null handling
def extract_nested_fields(df, fields):
    return [f.when(f.col(field).isNotNull(), f.col(field)).alias(new_name) for field, new_name in fields]


unzipped_df = process_gzip_file(args['source_bucket'], args['source_log_file'])

nested_fields = [
    ("eula.identifier", "eula_identifier"),
    ("eula.number", "eula_revision_number"),
    ("location.city", "location_city"),
    ("location.continent", "location_continent"),
    ("location.country", "location_country"),
    ("location.country_code", "location_country_code"),
    ("location.longitude", "location_longitude"),
    ("location.latitude", "location_latitude"),
    ("package.identifier", "package_identifier"),
    ("package.name", "package_name"),
    ("package.version", "package_version"),
    ("token.identifier", "token_identifier"),
    ("token.name", "token_name"),
    ("token.metadata", "token_metadata"),
    ("user.identifier", "user_identifier"),
    ("user.username", "user_username"),
    ("user_agent.browser", "user_agent_browser"),
    ("user_agent.device", "user_agent_device"),
    ("user_agent.os", "user_agent_os"),
    ("user_agent.raw", "user_agent_raw")
]

transformed_df = unzipped_df.select(
    # Convert "datetime" and "recorded" columns to timestamp
    f.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ssXXX").alias("datetime"),
    f.to_timestamp("recorded", "yyyy-MM-dd'T'HH:mm:ssXXX").alias("recorded"),

    f.to_json(f.col("package.tags")).alias("package_tags"),

    # Extract nested fields
    *extract_nested_fields(unzipped_df, nested_fields),

    # Include other columns, excluding nested structs
    *[f.col(col_name) for col_name in unzipped_df.columns if
      col_name not in ["datetime", "recorded", "eula", "location", "package", "token", "user", "user_agent"]],

    # Add partition columns based on datetime
    f.year("datetime").alias("year"),
    f.month("datetime").alias("month"),
    f.dayofmonth("datetime").alias("day")
)

transformed_df.write \
    .mode("append") \
    .partitionBy("year", "month", "day") \
    .parquet("s3a://" + args['target_bucket'] + "/")

job.commit()