import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import desc
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# Inspect Label Store 
gold_label_store_directory = "./datamart/gold/label_store/" 

folder_path = gold_label_store_directory
files_list = glob.glob(os.path.join(folder_path, '*'))

df = spark.read.parquet(*files_list)
print("row_count:", df.count())
df.show()

# print("Unique snapshot dates (Latest first):")
df.select("snapshot_date").distinct().orderBy(desc("snapshot_date")).show()

# set up config
model_train_date_str = "2024-12-01"
train_test_period_months = 12
oot_period_months = 5
train_test_ratio = 0.8

config = {}
config["model_train_date_str"] = model_train_date_str
config["train_test_period_months"] = train_test_period_months
config["oot_period_months"] =  oot_period_months
config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
config["train_test_ratio"] = train_test_ratio 

pprint.pprint(config)

# connect to label store
folder_path = "datamart/gold/label_store/"
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
label_store_sdf = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",label_store_sdf.count())

label_store_sdf.show()

# From Lab 5
# extract label store
labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))

print("extracted labels_sdf", labels_sdf.count(), config["train_test_start_date"], config["oot_end_date"])

# From Lab 5
# get features
feature_location = "data/feature_clickstream.csv"

# Load CSV into DataFrame - connect to feature store
features_store_sdf = spark.read.csv(feature_location, header=True, inferSchema=True)
print("row_count:",features_store_sdf.count())

features_store_sdf.show()

feat_filter_start = config["train_test_start_date"] - relativedelta(months=6)
feat_filter_end = config["oot_end_date"] - relativedelta(months=6)

features_sdf = features_store_sdf.filter(
    (col("snapshot_date") >= feat_filter_start) & 
    (col("snapshot_date") <= feat_filter_end)
)
print(features_sdf.count())

from pyspark.sql.functions import regexp_extract, to_date

labels_with_orig = labels_sdf.withColumn(
    "origination_date",
    to_date(regexp_extract(col("loan_id"), r"(\d{4}_\d{2}_\d{2})$", 1), "yyyy_MM_dd")
)
features_renamed = features_sdf.withColumnRenamed("snapshot_date", "origination_date")

data_pdf = labels_with_orig.join(features_renamed, on=["Customer_ID", "origination_date"], how="left").toPandas()
print(data_pdf.shape)
print(data_pdf.isnull().sum())

feat_filter_start = config["train_test_start_date"] - relativedelta(months=6)
feat_filter_end = config["oot_end_date"] - relativedelta(months=6)

features_sdf = features_store_sdf.filter(
    (col("snapshot_date") >= feat_filter_start) & 
    (col("snapshot_date") <= feat_filter_end)
)
print("extracted features_sdf", features_sdf.count(), feat_filter_start, feat_filter_end)

from pyspark.sql.functions import regexp_extract, to_date

labels_with_orig = labels_sdf.withColumn(
    "origination_date",
    to_date(regexp_extract(col("loan_id"), r"(\d{4}_\d{2}_\d{2})$", 1), "yyyy_MM_dd")
)
features_renamed = features_sdf.withColumnRenamed("snapshot_date", "origination_date")

data_pdf = labels_with_orig.join(features_renamed, on=["Customer_ID", "origination_date"], how="left").toPandas()
data_pdf

# From Lab5
# split data into train - test - oot
oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]

feature_cols = [fe_col for fe_col in data_pdf.columns if fe_col.startswith('fe_')]

X_oot = oot_pdf[feature_cols]
y_oot = oot_pdf["label"]
X_train, X_test, y_train, y_test = train_test_split(
    train_test_pdf[feature_cols], train_test_pdf["label"], 
    test_size= 1 - config["train_test_ratio"],
    random_state=88,     # Ensures reproducibility
    shuffle=True,        # Shuffle the data before splitting
    stratify=train_test_pdf["label"]           # Stratify based on the label column
)

print('X_train', X_train.shape[0])
print('X_test', X_test.shape[0])
print('X_oot', X_oot.shape[0])
print('y_train', y_train.shape[0], round(y_train.mean(),2))
print('y_test', y_test.shape[0], round(y_test.mean(),2))
print('y_oot', y_oot.shape[0], round(y_oot.mean(),2))

X_train

# set up standard scalar preprocessing
scaler = StandardScaler()

transformer_stdscaler = scaler.fit(X_train) # Q which should we use? train? test? oot? all?

# transform data
X_train_processed = transformer_stdscaler.transform(X_train)
X_test_processed = transformer_stdscaler.transform(X_test)
X_oot_processed = transformer_stdscaler.transform(X_oot)

print('X_train_processed', X_train_processed.shape[0])
print('X_test_processed', X_test_processed.shape[0])
print('X_oot_processed', X_oot_processed.shape[0])

pd.DataFrame(X_train_processed)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Train Logistic Regression
lr_clf = LogisticRegression(random_state=88, max_iter=1000)
lr_clf.fit(X_train_processed, y_train)

# Evaluate on train / test / oot
lr_train_auc = roc_auc_score(y_train, lr_clf.predict_proba(X_train_processed)[:, 1])
lr_test_auc  = roc_auc_score(y_test,  lr_clf.predict_proba(X_test_processed)[:, 1])
lr_oot_auc   = roc_auc_score(y_oot,   lr_clf.predict_proba(X_oot_processed)[:, 1])

print("=== Logistic Regression ===")
print(f"Train AUC : {lr_train_auc:.4f}  |  GINI: {round(2*lr_train_auc-1, 3)}")
print(f"Test  AUC : {lr_test_auc:.4f}  |  GINI: {round(2*lr_test_auc-1, 3)}")
print(f"OOT   AUC : {lr_oot_auc:.4f}  |  GINI: {round(2*lr_oot_auc-1, 3)}")

# --- save model artefact to model bank ---
model_name = model_name = "lr_model_2024_12_01.pkl"
model_bank_directory = "model_bank/"

if not os.path.exists(model_bank_directory):
    os.makedirs(model_bank_directory)

model_artefact = {
    "model": lr_clf,
    "preprocessing_transformers": {
        "stdscaler": transformer_stdscaler
    }
}

model_artefact_filepath = model_bank_directory + model_name
with open(model_artefact_filepath, 'wb') as f:
    pickle.dump(model_artefact, f)

print("Model saved to:", model_artefact_filepath)
