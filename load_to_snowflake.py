"""Loads outputs/clv_scored_output.csv into Snowflake and merges it into CLV_SCORED_OUTPUT.

Runs as a dry run unless Snowflake credentials are set as environment variables:
SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_WAREHOUSE.
Run sql/01_create_tables.sql and sql/02_merge_procedure.sql once before the first load.
"""
import os

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

scores = pd.read_csv("outputs/clv_scored_output.csv")
run_id = scores["RUN_ID"].iloc[0]

if "SNOWFLAKE_ACCOUNT" not in os.environ:
    print("DRY RUN: SNOWFLAKE_ACCOUNT is not set, nothing was sent to Snowflake.")
    print("Would load", len(scores), "rows of", run_id, "into CLV_DB.ML.CLV_SCORED_STAGE")
    print("Then call CLV_DB.ML.SP_MERGE_CLV_SCORES('" + run_id + "')")
    print(scores["RISK_BUCKET"].value_counts())
else:
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database="CLV_DB",
        schema="ML",
    )
    # write_pandas uploads the table as compressed Parquet and runs one COPY INTO,
    # much faster than sending rows one INSERT at a time
    success, chunks, rows, _ = write_pandas(conn, scores, "CLV_SCORED_STAGE")
    print("Staged", rows, "rows")

    cursor = conn.cursor()
    cursor.execute("CALL SP_MERGE_CLV_SCORES(%s)", (run_id,))
    print(cursor.fetchone()[0])
    conn.close()