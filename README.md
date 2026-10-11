# EverPure CLV Predictive Analytics (Case Study 2)

The Customer Lifetime Value portal projects 15-year net profit using a static 5% yearly churn assumption. This project tests that assumption against actual outcomes, trains two churn models (XGBoost and Random Forest), back-tests both against the baseline by cohort, and writes scored results with SHAP explanations to Snowflake.

**Main result:** actual 12-month churn is **15.8%**, not 5%. The baseline is too optimistic in every cohort and overstates the 15-year net profit projection by about 2x. The models remove most of that bias, but only because they learn the correct churn level: the features carry no detectable per-array signal (permutation test p = 0.13). Full numbers are in the [Back-Testing Validation Report](Backtest_Validation_Report.md).

## Deliverables

| Brief deliverable | File |
|---|---|
| 1. Documented Python ML codebase with feature pipelines | `CLV_Predictive_Analytics.ipynb` |
| 2. Back-Testing Validation Report | `Backtest_Validation_Report.md` |
| 3. Snowflake DDL and table insert scripts (`CLV_SCORED_OUTPUT`) | `sql/01_create_tables.sql`, `sql/02_merge_procedure.sql`, `sql/03_insert_clv_scored_output.sql`, `load_to_snowflake.py` |
| Technical handover documentation (rubric) | this README |

## Repository layout

```text
data/raw/Case_Study_2_CLV_Predictive_Analytics.csv   input data (2,000 arrays)
CLV_Predictive_Analytics.ipynb                       full pipeline: audit, features, models, back-tests, SHAP, export
outputs/clv_scored_output.csv                        2,000 scored arrays, same columns as CLV_SCORED_OUTPUT
outputs/cohort_backtest.csv                          bias by cohort, static 5% vs model
outputs/fig1_cohort_backtest.png                     churn by cohort chart
outputs/fig2_calibration.png                         calibration chart
outputs/fig3_shap_importance.png                     SHAP feature importance
sql/01_create_tables.sql                             output table, stage table, views
sql/02_merge_procedure.sql                           SP_MERGE_CLV_SCORES stored procedure
sql/03_insert_clv_scored_output.sql                  generated INSERT script for all 2,000 rows
load_to_snowflake.py                                 Python loader (write_pandas + MERGE), dry run by default
Backtest_Validation_Report.md                        back-testing report
requirements.txt                                     pinned library versions
```

## How to run

```bash
git clone https://github.com/SugarFreeBiskoot/everpure-clv-predictive-analytics.git
cd everpure-clv-predictive-analytics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `CLV_Predictive_Analytics.ipynb` and choose **Kernel → Restart Kernel and Run All Cells**. It runs in about 1 to 2 minutes and rewrites everything in `outputs/` and `sql/03_insert_clv_scored_output.sql`. All random steps use `SEED = 42`, so results are reproducible.

**macOS:** XGBoost needs the OpenMP library. If `import xgboost` fails, run `brew install libomp`.

**Loading to Snowflake**
1. Run `sql/01_create_tables.sql`, then `sql/02_merge_procedure.sql` (once).
2. Either run `sql/03_insert_clv_scored_output.sql`, or set `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD` and `SNOWFLAKE_WAREHOUSE` and run `python load_to_snowflake.py`. Without credentials the loader does a dry run and sends nothing.

## Data quality log

All checks are done in code (notebook cell 3) before any change. No rows were dropped.

| Issue | Rows | Treatment | Reason |
|---|---|---|---|
| `utilization_rate_pct` above 1.0 (max 1.45). The values are fractions, so 1.0 = 100% | 862 (43%) | `utilization_capped` = min(value, 1.0), plus `util_over_100_flag` and `util_excess` = max(value - 1.0, 0) | Capping alone would make 43% of rows identical. The flag and excess keep the information that capping would lose |
| Negative `historical_15yr_net_profit_usd` (min -93,780) | 58 | `profit_signed_log` = sign(x) * log(1 + abs(x)), plus `is_loss_making` flag | A plain log fails on negatives. The signed log keeps the sign and compresses the range |
| Missing `churn_event_next_12m` | 116 | Separated before any training. Scored by the final model, `HAS_LABEL = FALSE` in the output | An unknown outcome cannot be used for training or testing; filling it in would invent outcomes |
| `UNKNOWN` in `expansion_potential_signal` | 111 | Kept as its own one-hot category (`expansion_UNKNOWN`) | Filling with the most common value would invent data. Its churn rate (13.3%) is within noise of the other groups |
| Duplicates | 0 | None needed | Checked |
| `static_baseline_churn_rate` | all 0.05 | Used as the baseline in the back-test | Confirms the brief |

## Features

| Feature | Built from |
|---|---|
| `utilization_capped`, `util_over_100_flag`, `util_excess` | `utilization_rate_pct` |
| `profit_signed_log`, `is_loss_making` | `historical_15yr_net_profit_usd` |
| `capacity_tb` | `current_capacity_tb` |
| `days_to_contract_end` | `contract_end_date` minus the score date (2026-10-01, the earliest contract end in the data) |
| `expansion_Low`, `expansion_Medium`, `expansion_High`, `expansion_UNKNOWN` | `expansion_potential_signal` (one-hot) |

## Models and validation

- **XGBoost:** 300 trees, depth 3, learning rate 0.03, 80% row and column sampling. Kept shallow because there are only 298 churners.
- **Random Forest:** 500 trees, at least 20 arrays per leaf.
- No class weighting, because it would distort the probabilities that CLV needs.
- **Validation:** 5-fold `StratifiedGroupKFold` grouped by `account_id`, so both arrays of an account stay in the same fold. All back-test numbers are out-of-fold.
- **Champion:** Random Forest (Brier 0.1342 vs 0.1370 for XGBoost). It is retrained on all 1,884 labelled arrays and scores all 2,000.
- **Hyperparameters were not tuned.** The permutation test shows no signal to tune toward, and tuning on noise would overfit the validation folds.

**Why no survival model:** the brief allows Random Forest or a survival model. A survival model (Cox, Random Survival Forest) needs, for each array, the time until churn or until it stopped being observed. This data only has a 12-month yes/no outcome, so there is no duration to model. Using the contract end date as a duration would be inventing data. The 15-year view therefore uses S(t) = (1 - p)^t, labelled as an extrapolation.

## Scored output (`CLV_SCORED_OUTPUT`)

One row per array per score date per model version:
- IDs and run info: `ARRAY_ID`, `ACCOUNT_ID`, `SCORE_DATE`, `MODEL_NAME`, `MODEL_VERSION`, `RUN_ID`, `HAS_LABEL`
- Model features: the cleaned feature values
- `CHURN_PROB_12M`: Random Forest probability of churn in the next 12 months
- `RISK_BUCKET`: quartile of `CHURN_PROB_12M`. LOW 5.6% to 12.7%, MEDIUM 12.7% to 15.4%, HIGH 15.4% to 18.6%, VERY_HIGH 18.6% to 32.1%. Buckets are relative ranks; given the permutation result they should not drive targeting yet
- `SURVIVAL_PROB_15Y`: (1 - `CHURN_PROB_12M`)^15
- `HORIZON_BASIS`: `Y1_VALIDATED_Y2_15_EXTRAPOLATED`
- `TOP_DRIVERS` (VARIANT): the 3 features with the largest SHAP values for that array, in probability units, e.g. `[{"feature": "days_to_contract_end", "shap": 0.0312}, ...]`. The 4 expansion columns are summed into one `expansion_signal` driver

**Warehouse design:**
- The output table is never replaced, so past forecasts remain available for future back-tests.
- Loads land in a transient stage table. `SP_MERGE_CLV_SCORES(run_id)` MERGEs them into the output table, so re-running a load updates rows instead of duplicating them. Snowflake does not enforce primary keys, so the MERGE is what keeps the table unique.
- `V_CLV_TOP_DRIVERS` flattens the SHAP JSON into rows for BI tools, and `V_CLV_RUN_SUMMARY` summarises each run by bucket.

## Assumptions

Only where the maths requires one:
1. **Years 2 to 15 repeat the 12-month churn probability.** Only 12-month outcomes exist, so any 15-year figure needs this. Every row is labelled with it.
2. **Score date = earliest contract end date (2026-10-01).** A reference date is needed to turn dates into day counts.
3. **Cohorts are built from existing fields,** because the data has no product column. Every cut uses a natural boundary (year, 100%, profit sign) or equal-sized groups (capacity tiers).


## Monitoring and retraining

Labels arrive 12 months after scoring, so input drift is the early warning. Suggested triggers:
- **Feature or score drift:** Population Stability Index (PSI) above 0.2 between the training data and a new scoring run.
- **Calibration:** once a scored cohort's 12-month outcomes close, its mean prediction falls outside the 95% confidence interval of its actual churn rate.
- **Challenger:** the model's Brier score is worse than the constant observed-rate challenger on the latest closed cohort.
- **Business change:** a new product family, pricing or contract-term change, or new data sources becoming available.
- At least every 6 months otherwise.


## Limitations

The models produce a churn probability for every array and are accurate at portfolio and cohort level (bias within about +/-1.4 pp in 14 of 15 cohorts). At individual-array level they rank no better than chance, because the input fields carry no detectable churn signal (permutation test p = 0.13). The scores should therefore be used for forecasting churn levels, not for targeting individual arrays, until stronger features are available (see the report, Section 7).