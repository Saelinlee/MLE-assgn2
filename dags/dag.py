from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Root directory where the pipeline scripts live inside the Airflow container
SCRIPTS_DIR = "/opt/airflow/scripts"

# Make utils package importable from SCRIPTS_DIR at module load time
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# --- Datamart directory paths (relative to SCRIPTS_DIR) ---
BRONZE_LMS_DIR  = "datamart/bronze/lms/"
BRONZE_CLKS_DIR = "datamart/bronze/clks/"
BRONZE_ATTR_DIR = "datamart/bronze/attr/"
BRONZE_FIN_DIR  = "datamart/bronze/fin/"

SILVER_LMS_DIR  = "datamart/silver/lms/"
SILVER_CLKS_DIR = "datamart/silver/clks/"
SILVER_ATTR_DIR = "datamart/silver/attr/"
SILVER_FIN_DIR  = "datamart/silver/fin/"

GOLD_LABEL_DIR  = "datamart/gold/label_store/"
GOLD_ENG_DIR    = "datamart/gold/feature_store/eng/"
GOLD_RISK_DIR   = "datamart/gold/feature_store/cust_fin_risk/"

MODEL_NAME      = "lr_model_2024_12_01.pkl"


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# ---------------------------------------------------------------------------
# Helper: build a SparkSession for tasks running inside the Airflow worker
# ---------------------------------------------------------------------------
def _get_spark(app_name: str):
    import pyspark
    sys.path.insert(0, SCRIPTS_DIR)
    os.chdir(SCRIPTS_DIR)
    spark = (
        pyspark.sql.SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---------------------------------------------------------------------------
# Label Store callables  (Bronze → Silver → Gold for LMS / loan data)
# ---------------------------------------------------------------------------

def run_bronze_loan_table(**kwargs):
    from utils.data_processing_bronze_table import process_bronze_loan_table
    snapshot_date = kwargs["ds"]
    spark = _get_spark("bronze_loan")
    os.makedirs(BRONZE_LMS_DIR, exist_ok=True)
    process_bronze_loan_table(snapshot_date, BRONZE_LMS_DIR, spark)
    spark.stop()


def run_silver_loan_table(**kwargs):
    from utils.data_processing_silver_table import process_silver_loan_table
    snapshot_date = kwargs["ds"]
    spark = _get_spark("silver_loan")
    os.makedirs(SILVER_LMS_DIR, exist_ok=True)
    process_silver_loan_table(snapshot_date, BRONZE_LMS_DIR, SILVER_LMS_DIR, spark)
    spark.stop()


def run_gold_label_store(**kwargs):
    from utils.data_processing_gold_table import process_labels_gold_table
    snapshot_date = kwargs["ds"]
    spark = _get_spark("gold_labels")
    os.makedirs(GOLD_LABEL_DIR, exist_ok=True)
    process_labels_gold_table(
        snapshot_date, SILVER_LMS_DIR, GOLD_LABEL_DIR, spark, dpd=30, mob=6
    )
    spark.stop()


# ---------------------------------------------------------------------------
# Feature Store callables
#   bronze_table_1 = clickstream
#   bronze_table_2 = attributes
#   bronze_table_3 = financials
#   silver_table_1 = clickstream silver + attributes silver
#   silver_table_2 = financials silver
#   gold_feature_store = engagement gold + customer-financial-risk gold
# ---------------------------------------------------------------------------

def run_bronze_clickstream_table(**kwargs):
    from utils.data_processing_bronze_table import process_bronze_clickstream_table
    snapshot_date = kwargs["ds"]
    spark = _get_spark("bronze_clks")
    os.makedirs(BRONZE_CLKS_DIR, exist_ok=True)
    process_bronze_clickstream_table(snapshot_date, BRONZE_CLKS_DIR, spark)
    spark.stop()


def run_bronze_attributes_table(**kwargs):
    from utils.data_processing_bronze_table import process_bronze_attributes_table
    snapshot_date = kwargs["ds"]
    spark = _get_spark("bronze_attr")
    os.makedirs(BRONZE_ATTR_DIR, exist_ok=True)
    process_bronze_attributes_table(snapshot_date, BRONZE_ATTR_DIR, spark)
    spark.stop()


def run_bronze_financials_table(**kwargs):
    from utils.data_processing_bronze_table import process_bronze_financials_table
    snapshot_date = kwargs["ds"]
    spark = _get_spark("bronze_fin")
    os.makedirs(BRONZE_FIN_DIR, exist_ok=True)
    process_bronze_financials_table(snapshot_date, BRONZE_FIN_DIR, spark)
    spark.stop()


def run_silver_clickstream_and_attributes_table(**kwargs):
    """
    Processes silver clickstream and silver attributes in a single task so that
    both bronze_table_1 (clickstream) and bronze_table_2 (attributes) are
    guaranteed to be ready before this step runs.
    """
    from utils.data_processing_silver_table import (
        process_silver_clickstream_table,
        process_silver_attributes_table,
    )
    snapshot_date = kwargs["ds"]
    spark = _get_spark("silver_clks_attr")
    os.makedirs(SILVER_CLKS_DIR, exist_ok=True)
    os.makedirs(SILVER_ATTR_DIR, exist_ok=True)
    process_silver_clickstream_table(snapshot_date, BRONZE_CLKS_DIR, SILVER_CLKS_DIR, spark)
    process_silver_attributes_table(snapshot_date, BRONZE_ATTR_DIR, SILVER_ATTR_DIR, spark)
    spark.stop()


def run_silver_financials_table(**kwargs):
    from utils.data_processing_silver_table import process_silver_financials_table
    snapshot_date = kwargs["ds"]
    spark = _get_spark("silver_fin")
    os.makedirs(SILVER_FIN_DIR, exist_ok=True)
    process_silver_financials_table(snapshot_date, BRONZE_FIN_DIR, SILVER_FIN_DIR, spark)
    spark.stop()


def run_gold_feature_store(**kwargs):
    """
    Builds both gold feature tables:
      - Engagement features  (rolling 6-month clickstream window)
      - Customer financial-risk features
    """
    from utils.data_processing_gold_table import (
        process_fts_gold_engag_table,
        process_fts_gold_cust_risk_table,
    )
    snapshot_date = kwargs["ds"]
    spark = _get_spark("gold_feature_store")
    os.makedirs(GOLD_ENG_DIR, exist_ok=True)
    os.makedirs(GOLD_RISK_DIR, exist_ok=True)
    process_fts_gold_engag_table(snapshot_date, SILVER_CLKS_DIR, GOLD_ENG_DIR, spark)
    process_fts_gold_cust_risk_table(snapshot_date, SILVER_FIN_DIR, GOLD_RISK_DIR, spark)
    spark.stop()


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    "loan_default_pipeline",
    default_args=default_args,
    description="Monthly loan-default prediction pipeline (bronze → silver → gold → inference)",
    schedule_interval="0 0 1 * *",   # 1st of every month at 00:00 UTC
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # -----------------------------------------------------------------------
    # Label Store pipeline  (LMS data → dpd/mob labels)
    # -----------------------------------------------------------------------

    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    bronze_label_store = PythonOperator(
        task_id="bronze_label_store",
        python_callable=run_bronze_loan_table,
    )

    silver_label_store = PythonOperator(
        task_id="silver_label_store",
        python_callable=run_silver_loan_table,
    )

    gold_label_store = PythonOperator(
        task_id="gold_label_store",
        python_callable=run_gold_label_store,
    )

    label_store_completed = DummyOperator(task_id="label_store_completed")

    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed


    # -----------------------------------------------------------------------
    # Feature Store pipeline  (clickstream + attributes + financials)
    # -----------------------------------------------------------------------

    dep_check_source_data_bronze_1 = DummyOperator(task_id="dep_check_source_data_bronze_1")
    dep_check_source_data_bronze_2 = DummyOperator(task_id="dep_check_source_data_bronze_2")
    dep_check_source_data_bronze_3 = DummyOperator(task_id="dep_check_source_data_bronze_3")

    # bronze_table_1 = clickstream
    bronze_table_1 = PythonOperator(
        task_id="bronze_table_1",
        python_callable=run_bronze_clickstream_table,
    )

    # bronze_table_2 = attributes
    bronze_table_2 = PythonOperator(
        task_id="bronze_table_2",
        python_callable=run_bronze_attributes_table,
    )

    # bronze_table_3 = financials
    bronze_table_3 = PythonOperator(
        task_id="bronze_table_3",
        python_callable=run_bronze_financials_table,
    )

    # silver_table_1 = clickstream silver + attributes silver
    # (waits for both bronze_table_1 and bronze_table_2)
    silver_table_1 = PythonOperator(
        task_id="silver_table_1",
        python_callable=run_silver_clickstream_and_attributes_table,
    )

    # silver_table_2 = financials silver
    silver_table_2 = PythonOperator(
        task_id="silver_table_2",
        python_callable=run_silver_financials_table,
    )

    # gold_feature_store = engagement gold + cust-fin-risk gold
    # (waits for silver_table_1 and silver_table_2)
    gold_feature_store = PythonOperator(
        task_id="gold_feature_store",
        python_callable=run_gold_feature_store,
    )

    feature_store_completed = DummyOperator(task_id="feature_store_completed")

    dep_check_source_data_bronze_1 >> bronze_table_1 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_2 >> bronze_table_2 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_3 >> bronze_table_3 >> silver_table_2 >> gold_feature_store
    gold_feature_store >> feature_store_completed


    # -----------------------------------------------------------------------
    # Model Inference  (two models run in parallel after feature store ready)
    # -----------------------------------------------------------------------

    model_inference_start = DummyOperator(task_id="model_inference_start")

    # Model 1: logistic regression (lr_model_2024_12_01.pkl)
    model_1_inference = BashOperator(
        task_id="model_1_inference",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 model_inference.py '
            '--snapshotdate "{{ ds }}" '
            f'--modelname {MODEL_NAME}'
        ),
    )

    # Model 2: placeholder — swap in a second model artifact when available
    model_2_inference = DummyOperator(task_id="model_2_inference")

    model_inference_completed = DummyOperator(task_id="model_inference_completed")

    feature_store_completed >> model_inference_start
    model_inference_start >> model_1_inference >> model_inference_completed
    model_inference_start >> model_2_inference >> model_inference_completed


    # -----------------------------------------------------------------------
    # Model Monitoring  (checks prediction outputs for drift / quality)
    # -----------------------------------------------------------------------

    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    # Runs monitoring script: checks predictions from 6 months ago against labels now available
    model_1_monitor = BashOperator(
        task_id="model_1_monitor",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 model_monitoring.py '
            '--snapshotdate "{{ ds }}" '
            f'--modelname {MODEL_NAME}'
        ),
    )

    model_2_monitor = DummyOperator(task_id="model_2_monitor")

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")

    # Monitoring needs both inference output AND the label store for the current month
    model_inference_completed >> model_monitor_start
    label_store_completed     >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed
    model_monitor_start >> model_2_monitor >> model_monitor_completed


    # -----------------------------------------------------------------------
    # Model Auto-Training  (triggered when both feature store & labels ready)
    # -----------------------------------------------------------------------

    model_automl_start = DummyOperator(task_id="model_automl_start")

    # Placeholder: replace with a BashOperator calling a model-training script
    model_1_automl = DummyOperator(task_id="model_1_automl")
    model_2_automl = DummyOperator(task_id="model_2_automl")

    model_automl_completed = DummyOperator(task_id="model_automl_completed")

    feature_store_completed >> model_automl_start
    label_store_completed   >> model_automl_start
    model_automl_start >> model_1_automl >> model_automl_completed
    model_automl_start >> model_2_automl >> model_automl_completed
