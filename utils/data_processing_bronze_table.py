import os
import pandas as pd
import pyspark
from pyspark.sql.functions import col
from datetime import datetime


def _process_bronze_table(snapshot_date_str, csv_file_path, output_directory, partition_name, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + ' row count:', df.count())
    filepath = output_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)
    return df


def process_bronze_loan_table(snapshot_date_str, bronze_lms_directory, spark):
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-', '_') + '.csv'
    return _process_bronze_table(snapshot_date_str, "data/lms_loan_daily.csv", bronze_lms_directory, partition_name, spark)


def process_bronze_clickstream_table(snapshot_date_str, bronze_clks_directory, spark):
    partition_name = "bronze_clks_mthly_" + snapshot_date_str.replace('-', '_') + '.csv'
    return _process_bronze_table(snapshot_date_str, "data/feature_clickstream.csv", bronze_clks_directory, partition_name, spark)


def process_bronze_attributes_table(snapshot_date_str, bronze_attr_directory, spark):
    partition_name = "bronze_attr_mthly_" + snapshot_date_str.replace('-', '_') + '.csv'
    return _process_bronze_table(snapshot_date_str, "data/features_attributes.csv", bronze_attr_directory, partition_name, spark)


def process_bronze_financials_table(snapshot_date_str, bronze_fin_directory, spark):
    partition_name = "bronze_fin_mthly_" + snapshot_date_str.replace('-', '_') + '.csv'
    return _process_bronze_table(snapshot_date_str, "data/features_financials.csv", bronze_fin_directory, partition_name, spark)
