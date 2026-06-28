import argparse
import os
import glob
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql.functions import col
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score


# to call this script:
# python model_monitoring.py --snapshotdate 2024-01-01 --modelname lr_model_2024_12_01.pkl
#
# Logic: predictions were made at (snapshotdate - 6 months) for loans originated then.
# Labels for those loans are now available at snapshotdate (mob=6, dpd=30).
# PSI compares this month's score distribution vs. all prior months combined.


def compute_psi(reference_scores, current_scores, bins=10):
    breakpoints = np.linspace(0, 1, bins + 1)
    ref_pcts = np.histogram(reference_scores, bins=breakpoints)[0] / len(reference_scores)
    cur_pcts = np.histogram(current_scores, bins=breakpoints)[0] / len(current_scores)
    ref_pcts = np.where(ref_pcts == 0, 1e-6, ref_pcts)
    cur_pcts = np.where(cur_pcts == 0, 1e-6, cur_pcts)
    return float(np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts)))


def main(snapshotdate, modelname):
    print('\n\n---starting monitoring job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("model_monitoring") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    snapshot_date = datetime.strptime(snapshotdate, "%Y-%m-%d")
    model_name_stem = modelname[:-4]
    prediction_date = snapshot_date - relativedelta(months=6)
    prediction_date_str = prediction_date.strftime("%Y-%m-%d")

    print(f"Monitoring at: {snapshotdate}")
    print(f"Checking predictions made at: {prediction_date_str}")
    print(f"Against labels available at:  {snapshotdate}")

    pred_dir = f"datamart/gold/model_predictions/{model_name_stem}/"
    pred_file = f"{pred_dir}{model_name_stem}_predictions_{prediction_date_str.replace('-', '_')}.parquet"

    label_dir = "datamart/gold/label_store/"
    label_file = f"{label_dir}gold_label_store_{snapshotdate.replace('-', '_')}.parquet"

    results = {
        "snapshot_date": snapshotdate,
        "model_name": modelname,
        "prediction_date": prediction_date_str,
        "label_date": snapshotdate,
        "num_scored": np.nan,
        "num_matched": np.nan,
        "auc": np.nan,
        "gini": np.nan,
        "psi": np.nan,
    }

    # --- load predictions ---
    if not os.path.exists(pred_file):
        print(f"No predictions found at {pred_file}. Skipping.")
        predictions_pdf = None
    else:
        predictions_pdf = spark.read.parquet(pred_file).toPandas()
        results["num_scored"] = len(predictions_pdf)
        print(f"Loaded {len(predictions_pdf)} predictions")

    # --- load labels ---
    if not os.path.exists(label_file):
        print(f"No labels found at {label_file}. Skipping performance metrics.")
        labels_pdf = None
    else:
        labels_pdf = spark.read.parquet(label_file).toPandas()
        print(f"Loaded {len(labels_pdf)} labels")

    # --- performance metrics (AUC / Gini) ---
    if predictions_pdf is not None and labels_pdf is not None:
        merged = predictions_pdf.merge(labels_pdf[["Customer_ID", "label"]], on="Customer_ID", how="inner")
        results["num_matched"] = len(merged)
        if len(merged) > 0 and merged["label"].nunique() > 1:
            auc = roc_auc_score(merged["label"], merged["model_predictions"])
            results["auc"] = round(auc, 4)
            results["gini"] = round(2 * auc - 1, 4)
            print(f"AUC: {results['auc']}  Gini: {results['gini']}  N: {results['num_matched']}")
        else:
            print("Insufficient label variety to compute AUC.")

    # --- stability metric (PSI): current month vs. all prior months combined ---
    if predictions_pdf is not None:
        prior_files = [
            f for f in glob.glob(f"{pred_dir}*.parquet")
            if f != pred_file
        ]
        if prior_files:
            prior_scores = pd.concat([
                spark.read.parquet(f).toPandas()["model_predictions"]
                for f in prior_files
            ])
            results["psi"] = round(
                compute_psi(prior_scores.values, predictions_pdf["model_predictions"].values), 4
            )
            print(f"PSI vs prior months: {results['psi']}")
        else:
            print("No prior prediction batches found for PSI calculation.")

    # --- save monitoring results to gold table ---
    monitor_dir = "datamart/gold/model_monitoring/"
    os.makedirs(monitor_dir, exist_ok=True)

    partition_name = f"{model_name_stem}_monitoring_{snapshotdate.replace('-', '_')}.parquet"
    filepath = monitor_dir + partition_name
    spark.createDataFrame(pd.DataFrame([results])).write.mode("overwrite").parquet(filepath)
    print(f"Saved monitoring results to: {filepath}")

    spark.stop()
    print('\n\n---completed monitoring job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run monitoring job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
