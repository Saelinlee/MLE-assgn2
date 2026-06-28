import os
import glob
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType
from datetime import datetime

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

start_date_str = "2023-01-01"
end_date_str   = "2024-12-01"


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date   = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    current = datetime(start_date.year, start_date.month, 1)
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return dates


dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)


# ----------------------
# Build Bronze Tables
# ----------------------

bronze_steps = [
    (utils.data_processing_bronze_table.process_bronze_loan_table,         "datamart/bronze/lms/"),
    (utils.data_processing_bronze_table.process_bronze_clickstream_table,  "datamart/bronze/clks/"),
    (utils.data_processing_bronze_table.process_bronze_attributes_table,   "datamart/bronze/attr/"),
    (utils.data_processing_bronze_table.process_bronze_financials_table,   "datamart/bronze/fin/"),
]

for fn, directory in bronze_steps:
    os.makedirs(directory, exist_ok=True)
    for date_str in dates_str_lst:
        fn(date_str, directory, spark)


# ----------------------
# Build Silver Tables
# ----------------------

silver_lms_directory  = "datamart/silver/lms/"
silver_clks_directory = "datamart/silver/clks/"
silver_attr_directory = "datamart/silver/attr/"
silver_fin_directory  = "datamart/silver/fin/"

silver_steps = [
    (utils.data_processing_silver_table.process_silver_loan_table,
     "datamart/bronze/lms/", silver_lms_directory),
    (utils.data_processing_silver_table.process_silver_clickstream_table,
     "datamart/bronze/clks/", silver_clks_directory),
    (utils.data_processing_silver_table.process_silver_attributes_table,
     "datamart/bronze/attr/", silver_attr_directory),
    (utils.data_processing_silver_table.process_silver_financials_table,
     "datamart/bronze/fin/", silver_fin_directory),
]

for fn, bronze_dir, silver_dir in silver_steps:
    os.makedirs(silver_dir, exist_ok=True)
    for date_str in dates_str_lst:
        fn(date_str, bronze_dir, silver_dir, spark)


# ----------------------
# Build Gold Tables
# ----------------------

gold_clks_directory        = "datamart/gold/feature_store/eng/"
gold_fin_directory         = "datamart/gold/feature_store/cust_fin_risk/"
gold_label_store_directory = "datamart/gold/label_store/"

for directory in [gold_clks_directory, gold_fin_directory, gold_label_store_directory]:
    os.makedirs(directory, exist_ok=True)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_fts_gold_engag_table(date_str, silver_clks_directory, gold_clks_directory, spark)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_fts_gold_cust_risk_table(date_str, silver_fin_directory, gold_fin_directory, spark)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_lms_directory, gold_label_store_directory, spark, dpd=30, mob=6)


# ----------------------
# Inspect Gold Tables
# ----------------------

for label, folder_path in [("engagement", gold_clks_directory), ("cust_fin_risk", gold_fin_directory), ("label_store", gold_label_store_directory)]:
    files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.parquet(*files_list)
    print(f"\n[{label}] row_count: {df.count()}")
    df.show()


# ----------------------
# Stop Spark Session
# ----------------------

spark.stop()
print("Script finished execution. Stop Spark session.")
