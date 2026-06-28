import argparse
import os
import pickle
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from datetime import datetime


# to call this script: python model_inference.py --snapshotdate 2024-12-01 --modelname lr_model_2024_12_01.pkl

def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_artefact_filepath"] = "model_bank/" + modelname

    print(config)

    # --- load model artefact from model bank ---
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    print("Model loaded successfully! " + config["model_artefact_filepath"])


    # --- load feature store from silver clickstream (cleaned pipeline data) ---
    # Features must be as-of the loan origination date to avoid look-ahead bias
    silver_partition = "datamart/silver/clks/silver_clks_mthly_" + snapshotdate.replace("-", "_") + ".parquet"
    features_sdf = spark.read.parquet(silver_partition)
    print("extracted features_sdf", features_sdf.count(), snapshotdate)
    
    features_pdf = features_sdf.toPandas()


    # --- preprocess data for modeling ---
    # prepare X_inference
    feature_cols = [fe_col for fe_col in features_pdf.columns if fe_col.startswith('fe_')]
    X_inference = features_pdf[feature_cols]
    
    # apply transformer - standard scaler
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference = transformer_stdscaler.transform(X_inference)
    
    print('X_inference', X_inference.shape[0])


    # --- model prediction inference ---
    # load model
    model = model_artefact["model"]
    
    # predict model
    y_inference = model.predict_proba(X_inference)[:, 1]
    
    # prepare output
    y_inference_pdf = features_pdf[["Customer_ID","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    

    # --- save model inference to datamart gold table ---
    # create bronze datalake
    model_name_stem = config["model_name"][:-4]
    gold_directory = f"datamart/gold/model_predictions/{model_name_stem}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table - IRL connect to database to write
    partition_name = model_name_stem + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
