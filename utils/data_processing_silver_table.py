import os
import re
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType
from datetime import datetime


def process_silver_loan_table(snapshot_date_str, bronze_lms_directory, silver_lms_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_lms_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_clickstream_table(snapshot_date_str, bronze_clks_directory, silver_clks_directory, spark):
    partition_name = "bronze_clks_mthly_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_clks_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    column_type_map = {
        **{f"fe_{i}": IntegerType() for i in range(1, 21)},
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    for feature in [f"fe_{i}" for i in range(1, 21)]:
        df = df.withColumn(feature, F.when(F.col(feature) < 0, 0).otherwise(F.col(feature)))

    partition_name = "silver_clks_mthly_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_clks_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_attributes_table(snapshot_date_str, bronze_attr_directory, silver_attr_directory, spark):
    partition_name = "bronze_attr_mthly_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_attr_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    def clean_name(name):
        if name is None:
            return None
        return re.sub(r"[^A-Za-z .'-]", '', name).strip()

    df = df.withColumn("Name", F.udf(clean_name, StringType())(F.col("Name")))

    df = df.withColumn("Age", F.regexp_replace(F.col("Age").cast(StringType()), r"\D", ""))
    df = df.withColumn("Age", F.col("Age").cast(IntegerType()))
    df = df.withColumn("Age", F.when((F.col("Age") >= 0) & (F.col("Age") <= 100), F.col("Age")).otherwise(None))

    valid_ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
    df = df.withColumn("SSN", F.when(F.col("SSN").rlike(valid_ssn_pattern), F.col("SSN")).otherwise(None))

    df = df.withColumn("Occupation", F.when(F.col("Occupation") == '_______', None).otherwise(F.col("Occupation")))
    df = df.withColumn("Occupation", F.when(F.col("Occupation") == 'Media_Manager', 'Media Manager').otherwise(F.col("Occupation")))

    partition_name = "silver_attr_mthly_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_attr_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_financials_table(snapshot_date_str, bronze_fin_directory, silver_fin_directory, spark):
    partition_name = "bronze_fin_mthly_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_fin_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    cols_decimal3 = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
    ]
    for c in cols_decimal3:
        df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), r"[^\d.]", ""))
        df = df.withColumn(c, F.round(F.col(c).cast("double"), 3))
        df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))

    cols_integer = [
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan',
        'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Num_Credit_Inquiries', 'Interest_Rate'
    ]
    for c in cols_integer:
        df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), r"[^\d]", ""))
        df = df.withColumn(c, F.col(c).cast("int"))
        df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))

    def convert_to_months(s):
        if s is None:
            return None
        m = re.match(r"(\d+) Years and (\d+) Months", s)
        return int(m.group(1)) * 12 + int(m.group(2)) if m else None

    df = df.withColumn("Credit_History_Age", F.udf(convert_to_months, IntegerType())(F.col("Credit_History_Age")))

    cols_float = ['Changed_Credit_Limit', 'Credit_Utilization_Ratio']
    for c in cols_float:
        df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), r"[^\d.]", ""))
        df = df.withColumn(c, F.col(c).cast("double"))

    df = df.withColumn("Credit_Utilization_Ratio", F.when(F.col("Credit_Utilization_Ratio") < 0, None).otherwise(F.col("Credit_Utilization_Ratio")))

    outlier_caps = {
        'Num_Bank_Accounts': 10,
        'Num_Credit_Card': 10,
        'Interest_Rate': 34,
        'Num_of_Loan': 9,
        'Num_of_Delayed_Payment': 47,
        'Num_Credit_Inquiries': 26
    }
    for colname, cap in outlier_caps.items():
        df = df.withColumn(colname, F.when(F.col(colname) > cap, cap).otherwise(F.col(colname)))

    credit_mix_mapping = ['Bad', 'Standard', 'Good']
    df = df.withColumn("Credit_Mix", F.when(F.col("Credit_Mix").isin(credit_mix_mapping), F.col("Credit_Mix")).otherwise(None))
    df = df.withColumn("Credit_Mix",
        F.when(F.col("Credit_Mix") == "Bad", 0)
         .when(F.col("Credit_Mix") == "Standard", 1)
         .when(F.col("Credit_Mix") == "Good", 2)
         .cast("int"))

    df = df.withColumn("Payment_of_Min_Amount",
        F.when(F.col("Payment_of_Min_Amount") == "Yes", True)
         .when(F.col("Payment_of_Min_Amount") == "No", False)
         .otherwise(None))

    valid_pb_enums = [
        'High_spent_Small_value_payments', 'High_spent_Medium_value_payments',
        'High_spent_Large_value_payments', 'Low_spent_Small_value_payments',
        'Low_spent_Medium_value_payments', 'Low_spent_Large_value_payments'
    ]
    df = df.withColumn("Payment_Behaviour",
        F.when(F.col("Payment_Behaviour").isin(valid_pb_enums), F.col("Payment_Behaviour")).otherwise(None))
    df = df.withColumn("Payment_Behaviour",
        F.when(F.col("Payment_Behaviour") == "Low_spent_Small_value_payments", 0)
         .when(F.col("Payment_Behaviour") == "Low_spent_Medium_value_payments", 1)
         .when(F.col("Payment_Behaviour") == "Low_spent_Large_value_payments", 2)
         .when(F.col("Payment_Behaviour") == "High_spent_Small_value_payments", 3)
         .when(F.col("Payment_Behaviour") == "High_spent_Medium_value_payments", 4)
         .when(F.col("Payment_Behaviour") == "High_spent_Large_value_payments", 5)
         .cast("int"))

    df = df.withColumn("Num_Fin_Pdts", F.col("Num_Bank_Accounts") + F.col("Num_Credit_Card") + F.col("Num_of_Loan"))
    df = df.withColumn("Loans_per_Credit_Item", F.col("Num_of_Loan") / (F.col("Num_Bank_Accounts") + F.col("Num_Credit_Card") + F.lit(1)))
    df = df.withColumn("Debt_to_Salary", F.col("Outstanding_Debt") / (F.col("Monthly_Inhand_Salary") + F.lit(1)))
    df = df.withColumn("EMI_to_Salary", F.col("Total_EMI_per_month") / (F.col("Monthly_Inhand_Salary") + F.lit(1)))
    df = df.withColumn("Repayment_Ability", F.col("Monthly_Inhand_Salary") - F.col("Total_EMI_per_month"))
    df = df.withColumn("Loan_Extent", F.col("Delay_from_due_date") * F.col("Num_of_Loan"))

    partition_name = "silver_fin_mthly_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_fin_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
