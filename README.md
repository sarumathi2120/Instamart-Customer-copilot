# Instamart Customer Intelligence & Support Copilot

## Overview

This capstone uses synthetic Instamart-style data to build four connected parts: **A** data cleaning, SQL and analysis; **B** customer segmentation and lapse prediction; **C** support-ticket classification; and **D** policy search and a refund agent. Data and refunds are simulated, not real customer transactions.

## Reproduce

Run from the project folder (Python 3.11+, SQLite 3.25+):

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_data.py
python part_a.py
python validate.py
python part_b.py
python part_c.py
python part_d.py
```

The generator prints `customers=1200 products=40 orders_raw=13084 tickets=164`, and validation prints **ALL CHECKPOINTS PASSED**. Run scripts in order. `generate_data.py` and `mock_llm.py` are the supplied, unchanged scripts. The mock model runs offline without an API key. Dependencies are listed in `requirements.txt` (tested using Python 3.14).

## Part A

**Data audit and cleaning.** The raw datasets have 1,200 customers, 40 products, 13,084 order rows and 164 tickets.

| Finding | Raw count |
| --- | ---: |
| Duplicate order rows | 119 |
| Pending histories | 656 |
| Unique order IDs | 12,309 |
| Invalid/non-numeric order amounts | 120 |
| Negative / blank delivery times | 106 / 2,541 |
| Missing / invalid ages | 61 / 6 |
| City / nonblank gender spellings | 24 / 10 |
| Missing gender | 30 |

I standardised cities to **6** names and genders to male (**560**), female (**610**) or unknown (**30**). The **67** missing/invalid ages were filled with the corresponding city median; cleaned ages range **18–60**. Dates were parsed. Orders were cleaned by dropping **119** exact duplicates (**13,084 → 12,965**) and keeping the latest entry for each order ID (**12,965 → 12,309**). Missing amounts were calculated from price × quantity × (1 − discount/100). Invalid delivery times were replaced using completed orders' category medians; cancelled deliveries remain blank. All **40** products were preserved.

Final orders: **12,309**, including **11,110 completed**, **1,199 cancelled** and **0 pending**. There are **0** missing amounts or completed-order delivery times; cancelled orders have **1,199** missing delivery times. Completed-order revenue is **₹2,998,668.80**. Clean CSVs and `data/instamart.db` contain the results.

**SQL.** The seven queries are in `sql/queries.sql`; results are saved as `outputs/sql_q1.csv` to `sql_q7.csv`.

| Query | Main result |
| --- | --- |
| Q1 WHERE/HAVING | 30 customers |
| Q2 Three-table JOIN | 30 city/category groups; ₹2,998,668.80 total; Bengaluru Electronics ₹233,909.45 |
| Q3 Subquery | 11 above-average-price products |
| Q4 LEFT JOIN | 25 customers without completed orders |
| Q5 Two CTEs | 550 above-average customers; mean spend ₹2,552.06 |
| Q6 Window ranks | 1,175 customers; RANK differs from ROW_NUMBER in 8 rows |
| Q7 Running total | 183 days; final revenue ₹2,998,668.80 |

Top city spenders (customer ID, INR): Bengaluru **C0082, 12,856.20**; Chennai **C0930, 10,151.40**; Delhi **C0291, 13,742.60**; Hyderabad **C0743, 12,730.05**; Mumbai **C0693, 14,766.55**; Pune **C0426, 10,377.10**. A pandas pivot gave the same grand total as SQL. Bengaluru revenue was **₹523,643.40** (mean **₹276.47**, **1,894** orders); Mumbai **₹479,058.50** (mean **₹297.00**, **1,613** orders).

**EDA.** Order amounts are right-skewed: mean **₹269.91**, median **₹135.00**, skew **4.61**. Q1/Q3 are **₹64.80/₹297.00**; the 1.5×IQR outlier fences are **−₹283.50/₹645.30**, giving **897** high outliers and **0** low ones. Correlation of amount with discount is **−0.095** and with quantity **+0.087**; neither suggests a strong relationship or causation. Supporting files include `outputs/raw_audit.csv`, `order_stats.csv`, `city_summary.csv`, `city_category_pivot.csv` and `order_correlations.csv`.

## Part B

**RFM and clustering.** Using completed orders as of **2025-10-01**, there are **1,175** RFM customers; median recency is **19** days, average frequency **9.46**, average spend **₹2,552.06**. A K-means sweep gave:

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inertia | 1820.8 | 1268.2 | 743.1 | 617.4 | 503.2 | 408.5 | 366.5 |
| Silhouette | .465 | .445 | **.525** | .480 | .482 | .416 | .412 |

I chose **K=4** because it had the best silhouette and a clear elbow. Inertia always drops as K increases, so the smallest inertia alone is not a good choice.

| Segment (ID) | Customers | Mean recency | Frequency | Mean spend |
| --- | ---: | ---: | ---: | ---: |
| Occasional (0) | 436 | 28.9 | 3.8 | ₹1,057.62 |
| Premium (1) | 138 | 24.1 | 6.8 | ₹7,129.70 |
| Frequent Loyal (2) | 419 | 8.8 | 19.4 | ₹3,410.02 |
| At Risk (3) | 182 | 109.0 | 2.2 | ₹685.99 |

Suggested actions are reminders, premium service, loyalty rewards and re-engagement respectively. PCA explained variance is **0.634 + 0.218 = 0.852** for two components. Segments are in `outputs/customer_segments.csv`.

**Lapse prediction.** The prediction cutoff is **2025-09-01**: customers with no completed order during **September 1–30** are labelled lapsed. Only pre-cutoff transactions are used as predictive features, with preprocessing fitted on the training set. Cohort: **1,156** customers, **443** lapsed (**38.32%**); train/test **924/232** (**80/20**, stratified, seed **42**). The training scaler's frequency mean is **8.1147**, compared with cohort mean **8.0285**, confirming it was fitted on training rows only.

| Model | CV F1 | Test precision | Recall | F1 | Confusion matrix [TN,FP; FN,TP] |
| --- | ---: | ---: | ---: | ---: | --- |
| Majority baseline | — | 0 | 0 | 0 | [143,0;89,0] |
| Logistic Regression (C=100) | .651 | .6444 | .6517 | **.6480** | [111,32;31,58] |
| Random Forest (depth=5, leaf=10) | .684 | .6304 | .6517 | .6409 | [109,34;31,58] |
| Balanced logistic (training-only oversampling) | — | .5891 | .8539 | **.6972** | [90,53;13,76] |

Grid searches used **5-fold F1**: logistic `C={0.01,0.1,1,10,100}`, `max_iter=1000`; forest `max_depth={3,5,8}`, `min_samples_leaf={5,10,20}`, **100** trees. The selected saved model is **Logistic Regression**. Oversampling improved recall but reduced precision. The highest forest importances are frequency **.353**, monetary **.228**, recency **.097**, cancellation rate **.095**, discount **.051**. Logistic frequency coefficient is negative; its monetary coefficient is positive after controlling for other features, which is not proof of causality. Joblib reload reproduced test predictions exactly. The top **10** Premium customers are in `outputs/retention_list.csv`; **3/10** actually lapsed. That list is illustrative because the Premium segment uses the full time window.

## Part C

The support-ticket classifier uses the offline mock model. Privacy masking found **55 emails**, **91 phones**, and **132 order IDs** (**278** redactions). Empty tickets (**4**) are skipped; long tickets (**4**, estimated **734** tokens each) are flagged before calling the model because the limit is **400**. Output is checked against allowed categories; unclear responses go to human review. Retry logic uses up to **3** retries with exponential backoff around **0.05 × 2^n** seconds; **19** transient failures recovered in each full run.

| Prompt | Classified | Correct | Accuracy | Total tokens | Tokens/ticket |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 147 | 126 | **85.7%** | 9,088 | 61.82 |
| Few-shot | 147 | 147 | **100%** | 13,456 | 91.54 |

Both runs had **135** JSON classifications, **12** keyword fallbacks, **9** invalid outputs, **4** empty tickets and **4** long tickets. Zero-shot made **21** errors (**11** fraud and **10** technical tickets); few-shot used **29.72** extra tokens per classified ticket. Under the **6,000**-token budget demo, **53** tickets were marked `budget_exhausted`. Fraud routes to security; Premium non-fraud to the premium desk; other valid cases to support; invalid cases to human review. See `outputs/triage_results.csv`, `triage_budget_demo.csv` and `zero_shot_confusion.csv`.

## Part D

**Policy search.** Five sample policies were split into **40-word** TF-IDF chunks with **10-word** overlap: refund **4**, delivery **3**, escalation **3**, payment **3**, privacy **2** (**15** chunks). Cosine similarity picks the top **3** chunks for a grounded answer. TF measures term frequency; IDF lowers the weight of common terms.

| Setting | Correct document in top 3 | Correct answer | Unanswerable fallback |
| --- | --- | --- | --- |
| 10-word overlap | 12/12 | **12/12** | 2/2 |
| 0-word overlap | 12/12 | **10/12** | 2/2 |

Without overlap, facts crossing chunk boundaries were missed, including a delivery-time refund rule.

**Refund agent.** The agent uses `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`. Only completed orders qualify; delivery must exceed **45 minutes**; amounts above **₹1,000** need human approval. Test requests are assumed within **7 days**. No real refunds are issued.

| Case | Outcome | Amount | Model calls | Tokens |
| --- | --- | ---: | ---: | ---: |
| S1 | REFUND | ₹128.25 | 5 | 1,017 |
| S2 | REFUND | ₹135.00 | 5 | 1,017 |
| S3 | ESCALATE | ₹1,999.00 | 5 | 1,004 |
| S4 | DENIED | ₹0 | 4 | 701 |
| S5 | DENIED | ₹0 | 3 | 409 |
| S6 | NOT_FOUND | ₹0 | 2 | 120 |
| S7 | FALLBACK_MAX_ITERATIONS | ₹0 | 6 | 3,144 |

Safety checks also produced `FALLBACK_TOKEN_LIMIT` with a **200**-token cap and `FALLBACK_TIMEOUT` at **0** seconds. S7 stopped after **6** calls. A simulated S1 crash after the **second** tool resumed without repeating tools and finished with a **₹128.25** refund. Tool order: `get_order → search_policy → calculate_refund → issue_refund`. Traces are saved in `outputs/agent_runs.jsonl` and `state/`.

**Design choice.** ReAct made **5** model calls for S1, versus **3** for a cancelled order and **2** for a missing order. It can change its next step after seeing a tool result. Plan-and-execute could use fewer calls, but must replan when a tool gives an unexpected result. For refunds I prefer step-by-step checks and deterministic approval rules instead of relying only on a model.

## Limitations and Next Steps

This is synthetic data with a deterministic mock model. The **100%** few-shot result may not hold on real tickets. The retention list uses a full-period Premium flag and must not be treated as leakage-free. Policies, cutoff dates, payment operations and the refund ledger are simplified. Crash recovery demonstrates checkpointing, not production exactly-once guarantees. Next steps are stronger privacy and role controls, real-world validation, policy-search improvements, monitoring, idempotent payment handling and human approval for sensitive decisions.
