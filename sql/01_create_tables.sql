-- EverPure CLV: Snowflake objects for the scored churn model output
-- Run once, before the first load.

CREATE DATABASE IF NOT EXISTS CLV_DB;
CREATE SCHEMA IF NOT EXISTS CLV_DB.ML;
USE SCHEMA CLV_DB.ML;

-- Final table read by the CLV portal: one row per array, per score date, per model version.
-- Old runs are never overwritten, so every past forecast stays available for back-testing.
CREATE TABLE IF NOT EXISTS CLV_DB.ML.CLV_SCORED_OUTPUT (
    ARRAY_ID              VARCHAR(20)    NOT NULL,
    ACCOUNT_ID            VARCHAR(20)    NOT NULL,
    SCORE_DATE            DATE           NOT NULL,
    MODEL_NAME            VARCHAR(50)    NOT NULL,
    MODEL_VERSION         VARCHAR(20)    NOT NULL,
    RUN_ID                VARCHAR(50)    NOT NULL,
    HAS_LABEL             BOOLEAN,                 -- FALSE = one of the 116 arrays with no known outcome

    -- model features (cleaned)
    UTILIZATION_CAPPED    NUMBER(5,2),
    UTIL_OVER_100_FLAG    NUMBER(1,0),
    UTIL_EXCESS           NUMBER(5,2),
    CURRENT_CAPACITY_TB   NUMBER(10,0),
    NET_PROFIT_15YR_USD   NUMBER(18,2),
    IS_LOSS_MAKING        NUMBER(1,0),
    DAYS_TO_CONTRACT_END  NUMBER(6,0),
    EXPANSION_SIGNAL      VARCHAR(10),

    -- scores
    CHURN_PROB_12M        NUMBER(6,4)    NOT NULL,
    BASELINE_CHURN_RATE   NUMBER(6,4),
    RISK_BUCKET           VARCHAR(10)    NOT NULL, -- quartile of CHURN_PROB_12M: LOW, MEDIUM, HIGH, VERY_HIGH
    SURVIVAL_PROB_15Y     NUMBER(6,4),             -- 15-year renewal probability
    HORIZON_BASIS         VARCHAR(40),             -- year 1 validated, years 2-15 extrapolated

    -- SHAP explainability: top 3 drivers, e.g. [{"feature": "capacity_tb", "shap": 0.021}, ...]
    TOP_DRIVERS           VARIANT,

    LOADED_AT             TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_CLV_SCORED_OUTPUT PRIMARY KEY (ARRAY_ID, SCORE_DATE, MODEL_VERSION)
)
CLUSTER BY (SCORE_DATE, RISK_BUCKET);   -- the portal filters by date and bucket


-- Landing table for one scoring run. TRANSIENT = no fail-safe storage cost for temporary data.
-- SHAP drivers arrive as JSON text and are turned into VARIANT by the MERGE procedure.
CREATE TRANSIENT TABLE IF NOT EXISTS CLV_DB.ML.CLV_SCORED_STAGE (
    ARRAY_ID              VARCHAR(20),
    ACCOUNT_ID            VARCHAR(20),
    SCORE_DATE            DATE,
    MODEL_NAME            VARCHAR(50),
    MODEL_VERSION         VARCHAR(20),
    RUN_ID                VARCHAR(50),
    HAS_LABEL             BOOLEAN,
    UTILIZATION_CAPPED    NUMBER(5,2),
    UTIL_OVER_100_FLAG    NUMBER(1,0),
    UTIL_EXCESS           NUMBER(5,2),
    CURRENT_CAPACITY_TB   NUMBER(10,0),
    NET_PROFIT_15YR_USD   NUMBER(18,2),
    IS_LOSS_MAKING        NUMBER(1,0),
    DAYS_TO_CONTRACT_END  NUMBER(6,0),
    EXPANSION_SIGNAL      VARCHAR(10),
    CHURN_PROB_12M        NUMBER(6,4),
    BASELINE_CHURN_RATE   NUMBER(6,4),
    RISK_BUCKET           VARCHAR(10),
    SURVIVAL_PROB_15Y     NUMBER(6,4),
    HORIZON_BASIS         VARCHAR(40),
    TOP_DRIVERS_JSON      VARCHAR
);


-- One row per array and driver, so BI tools can read SHAP without parsing JSON
CREATE OR REPLACE VIEW CLV_DB.ML.V_CLV_TOP_DRIVERS AS
SELECT
    o.ARRAY_ID,
    o.SCORE_DATE,
    o.MODEL_VERSION,
    d.INDEX + 1               AS DRIVER_RANK,
    d.VALUE:feature::STRING   AS FEATURE,
    d.VALUE:shap::FLOAT       AS SHAP_VALUE
FROM CLV_DB.ML.CLV_SCORED_OUTPUT o,
     LATERAL FLATTEN(INPUT => o.TOP_DRIVERS) d;


-- Run summary per risk bucket: model churn vs the static 5% assumption
CREATE OR REPLACE VIEW CLV_DB.ML.V_CLV_RUN_SUMMARY AS
SELECT
    SCORE_DATE,
    MODEL_VERSION,
    RISK_BUCKET,
    COUNT(*)                 AS ARRAYS,
    AVG(CHURN_PROB_12M)      AS AVG_CHURN_PROB_12M,
    AVG(BASELINE_CHURN_RATE) AS BASELINE_CHURN_RATE,
    AVG(SURVIVAL_PROB_15Y)   AS AVG_SURVIVAL_PROB_15Y
FROM CLV_DB.ML.CLV_SCORED_OUTPUT
GROUP BY SCORE_DATE, MODEL_VERSION, RISK_BUCKET;