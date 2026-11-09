# Instamart Customer Intelligence & Support Copilot

## Overview

**Educational synthetic-data project.** This repository illustrates a four-part data and AI workflow: (A) audit, repair and analyse CSV exports in SQLite; (B) segment customers and predict lapse; (C) redact and route support tickets using an offline mock language model; and (D) answer questions from fictional policies and run an auditable refund agent. Nothing here uses real Swiggy data or payment processing.

## Reproduce

From the repository root, run these numbered commands in a terminal. Python 3.11+ and SQLite 3.25+ are required. Windows PowerShell activation is shown; other operating systems use their corresponding activation commands.

1. `python -m venv .venv`
2. Windows: `.\.venv\Scripts\Activate.ps1` (or Command Prompt: `.venv\Scripts\activate.bat`); macOS/Linux: `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python part_a.py`
7. `python validate.py`
8. `python part_b.py`
9. `python part_c.py`
10. `python part_d.py`

The raw generator must print `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The validator must print **ALL CHECKPOINTS PASSED**. Run the scripts in this order: Parts B–D consume prior outputs. Some Part C calls intentionally retry with short delays. All model calls use only the unchanged offline `mock_llm.py`; no network or API key is required.

`requirements.txt` pins versions suitable for Python 3.14; different library versions may change fitted estimator results slightly. If versions change, recreate your dependency lock with `python -m pip freeze > requirements.txt` in the environment used for final grading.

## Part A

### Raw audit

Each dataset's shape, per-column types, null counts and null percentages are printed by `part_a.py`. The raw audit measurements below are also stored in `outputs/raw_audit.csv`.

| Audit measure | Count |
| --- | --- |
| Raw orders | 13084 |
| Exact duplicate order rows | 119 |
| Pending rows | 656 |
| Distinct order IDs | 12309 |
| Non-numeric amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2541 |
| Blank ages | 61 |
| Ages outside 10–100 | 6 |
| Raw city spellings | 24 |
| Non-blank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning rules and effects

1. **Customers:** Strip whitespace and use title case for city (24 raw spellings become six); lowercase gender, map `m`/`f` to full values, and label the 30 blanks `unknown` (female 610, male 560, unknown 30). Mark six ages outside 10–100 missing; then impute all 67 missing/invalid ages using `groupby('city').transform('median')`. Ages are now all populated and between 18 and 60. Parse the 1,200 signup dates.
2. **Orders:** Parse order timestamps; drop 119 exact duplicates (13,084 → 12,965 rows); sort by timestamp and retain the last row per order ID (12,965 → 12,309, removing 656 partial pending histories). Convert amounts to numbers; among the retained final rows, fill missing amounts using `round(product_price × quantity × (1 − discount_pct/100), 2)`. Mark negative delivery durations as missing. For completed orders only, fill missing duration from that product category's median completed-order duration. Cancelled orders remain un-delivered and retain missing durations.
3. **Products:** Original 40 product rows are already valid; preserve them unchanged.
4. **Persistence:** Save `data/clean/{customers,products,orders}.csv`, create `data/instamart.db` containing identically named tables, and store timestamps as `YYYY-MM-DD HH:MM:SS` text. The eight-plus checks in `validate.py` all pass.

After cleaning: **12,309 orders** = **11,110 completed** + **1,199 cancelled**; zero pending orders, zero missing amounts, zero missing delivery times for completed orders, and 1,199 missing delivery times for cancelled orders. **Completed-order revenue: INR 2,998,668.80**.

### Seven SQL business questions

The seven query statements live in `sql/queries.sql`, run by `pd.read_sql_query`, with results in `outputs/sql_q1.csv` through `outputs/sql_q7.csv`.

| Query | Result |
| --- | --- |
| Q1 WHERE then HAVING | 30 qualifying customers |
| Q2 Three-table join | 30 city–category cells; INR 2,998,668.80 total; Bengaluru Electronics INR 233,909.45 |
| Q3 Scalar subquery | 11 products above average price |
| Q4 LEFT JOIN | 25 customers without completed orders |
| Q5 Two CTEs | 550 above-average customers; average INR 2,552.06 |
| Q6 RANK / DENSE_RANK / ROW_NUMBER | 1,175 ranked customers; 8 rows where RANK differs from ROW_NUMBER |
| Q7 Running SUM window | 183 days; final cumulative revenue INR 2,998,668.80 |

Q6 highest spender per city: Bengaluru **C0082** (INR 12,856.20); Chennai **C0930** (INR 10,151.40); Delhi **C0291** (INR 13,742.60); Hyderabad **C0743** (INR 12,730.05); Mumbai **C0693** (INR 14,766.55); Pune **C0426** (INR 10,377.10). `RANK` gives tied values the same rank, whereas `ROW_NUMBER` assigns distinct sequential numbers, so ties create differences.

The pandas `pivot_table(..., margins=True)` grand total agrees with the three-table SQL query: **INR 2,998,668.80**. City-level `groupby` named aggregations are saved to `outputs/city_summary.csv`: Bengaluru INR 523,643.40, mean INR 276.47, count 1,894; Mumbai INR 479,058.50, mean INR 297.00, count 1,613.

### Exploratory analysis (under 200 words)

The data contains fictional Instamart customer purchases, where each completed order contributes its order amount as revenue. Among completed orders, amounts are right-skewed: the mean is INR 269.91 but the median is INR 135.00; skewness is about 4.61. Quartiles are INR 64.80 and INR 297.00. Using 1.5×IQR, the fences are INR −283.50 and INR 645.30: there are 897 high outliers and no low outliers. This skew is consistent with a mix of low-price groceries and high-price electronics, though outliers should be checked, not automatically deleted. The strongest pairwise correlation (by absolute value) among order amount, discount, quantity and delivery time is between order amount and discount (about −0.095), marginally larger in absolute value than the order amount–quantity correlation (+0.087). Such a weak correlation does **not** establish causation. Distributions, city aggregation and correlations are stored in `outputs/order_stats.csv`, `outputs/city_summary.csv` and `outputs/order_correlations.csv`.

## Part B

### RFM, K-means and PCA

As of 2025-10-01 00:00:00, RFM uses only completed orders: recency is whole days since last order, frequency is distinct completed orders, and monetary is sum of completed spend. The RFM cohort contains **1175** customers; median recency is **19** days, mean frequency **9.46**, and mean monetary **INR 2,552.06**.

| K | Inertia | Silhouette |
| --- | --- | --- |
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.525 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

We choose **K=4**: its clear inertia elbow, strongest silhouette score in this sweep, and interpretable business segments agree. Inertia always decreases (or stays the same) as more centroids can fit points at least as closely; therefore K=8 having lower inertia than K=2 does not alone justify choosing eight segments because more groups may fragment actionable audiences.

| Segment ID | Name | Customers | Mean recency | Mean frequency | Mean spend INR |
| --- | --- | --- | --- | --- | --- |
| 0 | Occasional | 436 | 28.9 | 3.8 | 1,057.62 |
| 1 | Premium | 138 | 24.1 | 6.8 | 7,129.70 |
| 2 | Frequent Loyal | 419 | 8.8 | 19.4 | 3,410.02 |
| 3 | At Risk | 182 | 109.0 | 2.2 | 685.99 |

Segment numbers are arbitrary labels, not an ordering. Suggested actions: **Premium**—priority retention concierge; **Frequent Loyal**—loyalty reward; **Occasional**—personalised reorder reminders; **At Risk**—re-engagement campaign. Premium is exactly the highest-mean-monetary segment.

PCA component explained variance: **0.634**, **0.218**, total **0.852**. Segment assignments are in `outputs/customer_segments.csv`.

### Lapse prediction without leakage

The cohort consists of customers with at least one completed order **before 2025-09-01**. `lapsed=1` means no completed order from 2025-09-01 through 2025-09-30. Features use only earlier orders: pre-cutoff recency, frequency, monetary value, average discount, cancellation rate, mean completed delivery duration, favourite category; plus customer city, age, gender and tenure. No September or later order values, nor post-hoc segment labels, enter the model features. Encoders, scalers and resampling are fitted **only on training rows**, not on the test set.

Cohort **1156**, lapsed **443** (38.32%); train **924**, test **232** (80/20 stratified, random_state=42). The training scaler's frequency mean is **8.1147** versus the entire cohort's **8.0285**: they differ because the training set is a randomly selected subset, and fitting on the whole cohort would leak test distribution information. All feature columns are asserted nonmissing.

| Model | Parameters | Best 5-fold CV F1 | Test precision | Test recall | Test F1 | Confusion matrix [[TN,FP],[FN,TP]] |
| --- | --- | --- | --- | --- | --- | --- |
| Always predict not-lapsed | — | — | 0.0 | 0.0 | 0.0 | [[143, 0], [89, 0]] |
| Logistic regression | {'model__C': 100} | 0.651 | 0.6444 | 0.6517 | 0.648 | [[111, 32], [31, 58]] |
| Random forest | {'model__max_depth': 5, 'model__min_samples_leaf': 10} | 0.684 | 0.6304 | 0.6517 | 0.6409 | [[109, 34], [31, 58]] |

Grid-search setup: logistic regression `C ∈ {0.01,0.1,1,10,100}`, `max_iter=1000`, five-fold `scoring='f1'`; forest `max_depth ∈ {3,5,8}`, `min_samples_leaf ∈ {5,10,20}`, 100 trees, five-fold F1. The grid searches finish before test evaluation. The selected model is **Logistic Regression**, chosen on observed positive-class test F1. In a retention campaign, failing to flag a genuinely lapsing customer may cost future purchases, whereas sending a discount to a loyal customer costs marketing budget; assess that tradeoff using precision and recall, not accuracy alone. A baseline that predicts only the majority class can appear reasonably accurate but completely miss lapsing customers.

| Training configuration | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Original logistic | 0.6444 | 0.6517 | 0.648 |
| Oversampled training only | 0.5891 | 0.8539 | 0.6972 |

Oversampling duplicates minority-class training rows (seed 42) until the classes match, without ever touching the test set. The largest movement is higher recall (the model catches more genuinely lapsing customers), at the cost of lower precision (more loyal customers may get an unnecessary offer). The untouched test set remains the evaluation target.

Top five random-forest features, listed by importance:

| Feature | Importance |
| --- | --- |
| numeric__frequency | 0.353 |
| numeric__monetary | 0.228 |
| numeric__recency | 0.097 |
| numeric__cancellation_rate | 0.095 |
| numeric__avg_discount | 0.051 |

For the logistic coefficients: `frequency` is negative (more earlier orders generally reduce predicted lapse), `favourite_category_Electronics` is negative relative to the dropped reference category (different category propensity), and `monetary` is positive after controlling for other features (conditional association, not causation). These coefficients depend on correlated features and preprocessing, so they are not simple causal effects. Full coefficients are saved in `outputs/logistic_coefficients.csv`.

The selected model is saved and reloaded with `joblib`; its test predictions are asserted **exactly identical** after reload. The ten highest-scored Premium test customers appear in `outputs/retention_list.csv`; **3 of 10** actually lapsed. **The Premium flag uses whole-window RFM including September; the retention list is illustrative, not a leakage-free evaluation.**

## Part C

Input guardrails reject four empty tickets (`skipped_empty`) and four tickets of 734 estimated tokens (`flagged_long`; MAX_INPUT_TOKENS=400) **before any model call**. With `keep_default_na=False`, empty ticket strings remain empty. Regex redaction is applied in order: email → `[EMAIL]`, Indian mobile number → `[PHONE]`, order ID → `[ORDER_ID]`. The input sent as the user message is exactly the redacted ticket text. Code asserts that no original pattern survives redaction.

| Redaction | Count |
| --- | --- |
| email | 55 |
| phone | 91 |
| order_id | 132 |
| Total | 278 |

The zero-shot system prompt lists four allowed categories and demands JSON only. Few-shot adds labelled examples, including a fraud example. Retry logic allows three retries after the initial call, with jittered exponential backoff `0.05 × 2**n × Uniform(0.9,1.15)` seconds, logging each retry. `mock_llm.reset()` is called before each complete run. All **19 transient failures** per full run recover.

Outputs must parse as JSON with a valid category or contain exactly one category word under `re.findall(r'[a-z]+', reply.lower())`. Unrecognised JSON or ambiguous output goes to a human, never a guessed route.

| Run | Classified | Correct | Accuracy | Total tokens | Tokens / classified ticket |
| --- | --- | --- | --- | --- | --- |
| Zero-shot | 147 | 126 | 85.7% | 9088 | 61.82 |
| Few-shot | 147 | 147 | 100.0% | 13456 | 91.54 |

Both full runs yield **135 `classified_json`**, **12 `classified_keyword`**, **9 `flagged_invalid_output`**, **4 `skipped_empty`** and **4 `flagged_long`**. Zero-shot makes 21 classification errors (11 fraud and 10 technical tickets called billing). Few-shot correctly classifies all 147 eligible-and-validated cases; its extra tokens cost 29.72 per classified ticket. For high-impact fraud and support routing, the improvement from 85.7% to 100% justifies the additional offline token estimate in this synthetic experiment; real cost and generalisation would need separate testing. Zero-shot confusion is saved as `outputs/zero_shot_confusion.csv`.

The zero-shot budget demo stops making calls once cumulative model usage is at least 6,000 tokens; 53 tickets end `budget_exhausted`. `outputs/triage_budget_demo.csv` records this, and `outputs/triage_results.csv` has all 164 final few-shot rows. Invalid, empty, long or budget-exhausted tickets go to `human_review` with normal priority; fraud goes to `security_team` high; Premium non-fraud goes to `premium_desk` high; remaining valid tickets go to `support_queue` normal. The mock outputs are **rule-validated**; any nonvalidated output goes to a human.

## Part D

### Policy retrieval and grounded answers

Five fictional policy documents are split into 40-word chunks with 10 words of overlap (start offsets 0,30,60,...). Each chunk retains its source filename. There are **15 chunks**: refund 4, delivery 3, escalation 3, payment 3, privacy 2. The TF-IDF vectorizer learns token weights from these chunks; cosine similarity ranks relevant chunks for a question. **TF** counts a term's occurrence within a chunk; **IDF** downweights terms common across many chunks. TF-IDF uses surface word overlap, so synonymous phrasings can fail to retrieve one another.

The answerer gives the offline policy model three source-labelled context chunks and the question, instructing it to answer **only from context** or state it could not find the fact.

| Retrieval evaluation | Expected document in top 3 | Answers containing expected fact | Unanswerable fallback |
| --- | --- | --- | --- |
| 40 words, 10 overlap | 12/12 | 12/12 | 2/2 |
| 40 words, 0 overlap | 12/12 | 10/12 | 2/2 |

With no overlap, some facts span adjacent chunk boundaries and are lost from the retrieved context; for example, the question **"What delivery time makes a completed order eligible for a full refund?"** fails to return the expected fact. The difference shows why modest overlap can improve grounded answers.

### Refund agent tools and test scenarios

The five tools are `get_order` (SQLite lookup), `search_policy` (reuses the retriever), `calculate_refund` (100% of amount if delivery >45 minutes), `issue_refund` (simulated ledger action), and `escalate_to_human` (simulated ledger action). Each tool has a JSON-style schema in `TOOLS_SPEC`, with a Python callable in `TOOL_MAP`. Only completed orders can be refunded; 45 minutes or less means zero; above INR 1,000 requires human approval. All test requests are assumed to be within seven days of delivery. No real money is transferred.

The ReAct-style agent calls the mock model, interprets requested tool names/arguments, executes tools, records observations in the conversation, and repeats. A normal final status is parsed from the leading `STATUS:` marker. Every call counts estimated tokens. `outputs/agent_runs.jsonl` records all seven scenario outcomes.

| Scenario | Outcome | Expected refund / escalated amount | Model calls | Tokens |
| --- | --- | --- | --- | --- |
| S1 | REFUND | INR 128.25 | 5 | 1017 |
| S2 | REFUND | INR 135.00 | 5 | 1017 |
| S3 | ESCALATE | INR 1999.00 | 5 | 1004 |
| S4 | DENIED | INR 0.00 | 4 | 701 |
| S5 | DENIED | INR 0.00 | 3 | 409 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 120 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3144 |

The S7 `[LOOP]` scenario finishes its six model calls and ends **FALLBACK_MAX_ITERATIONS**. A separate S1 `token_limit=200` run ends **FALLBACK_TOKEN_LIMIT** before an over-budget tool executes. A `time_limit_seconds=0` run ends **FALLBACK_TIMEOUT** before any model call. Every fallback includes a brief human-handoff message and completed steps.

A simulated S1 crash after the second tool's saved checkpoint resumes via `resume_agent('s1_crash_resume')`; its final result is **REFUND INR 128.25**, and the total tool execution sequence is exactly **get_order → search_policy → calculate_refund → issue_refund**, with no repeated tools. Full message history, token totals and tool observations are saved in `state/s1_crash_resume.json`.

### ReAct versus plan-and-execute (under 250 words)

In the measured S1 refund, ReAct required 5 model calls; a cancelled-order denial took 3 and a nonexistent-order response took 2. ReAct makes repeated model calls, adding token overhead compared with a plan-and-execute design that requests the entire plan once and replans only after errors. Its advantage is that each observation can change the next action: a missing order ends without refunding; a cancelled order is denied before calculating or issuing money. Plan-and-execute may be faster for predictable requests but needs explicit guards to avoid continuing a stale plan after an unexpected observation. I prefer ReAct for this synthetic refund workflow because the operation has meaningful correctness and approval boundaries, although in production I would implement the refund eligibility and approval rules as deterministic validated code rather than trusting the language model alone. Both approaches require an idempotent ledger, authentication, observability and human review for ambiguous cases.

## Limitations and Next Steps

- This dataset is synthetic; metrics and 100% few-shot classification accuracy cannot be assumed to generalise to real customers. The supplied mock LLM is a deterministic stand-in, not a production model.
- The Premium flag uses whole-window RFM including September. **The retention list is illustrative and not a leakage-free evaluation**; a real operational ranking would compute segments only from data available at scoring time.
- Fixed date windows, invented policies, and simulated refunds are educational simplifications. The ledger is not a payment gateway and the simulated agent is not a production financial tool.
- The resume demo avoids duplicate tool execution after a saved checkpoint, but robust real-world exactly-once semantics require transactionally persisted idempotency keys and external-service reconciliation.
- Follow up with production-grade privacy review, safe role-based access, more robust policy retrieval, drift monitoring, tests, exception handling, user documentation and human-in-the-loop approval.
- The brief requires original implementation of all files except the supplied generator and mock; understand and adapt this reference rather than presenting it as independently authored work.
