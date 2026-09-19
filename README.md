# AttritionRadar
### Employee Attrition & Workforce Cost Dashboard

## 🚀 Live Dashboard

**[Open the AttritionRadar live Streamlit dashboard ↗](https://harsha-attritionradar.streamlit.app)**

Try the interactive dashboard directly in your browser. The GitHub repository contains the complete source code, while the Streamlit deployment is the live, interactive version for portfolio reviewers and hiring managers.

> **How to use this document:** Paste this whole file into ChatGPT as your
> first message and ask it to build the project exactly as specified below,
> file by file, in the order given in Section 11.

## 1. Business Problem
Unplanned employee attrition is expensive — recruiting, onboarding, and lost
productivity often cost 50–150% of a departing employee's salary. HR/People
teams need to know *which* segments (department, level, tenure band) are
highest-risk *before* they lose people, and roughly what that risk is
costing, so retention budget goes where it matters most.

**Business questions this project answers:**
- Which departments/levels have the highest voluntary turnover?
- Which active employees are at the highest predicted attrition risk?
- What's the estimated annual cost of that attrition exposure?

## 2. Business Impact
Converts an HR headcount report into a $-quantified, prioritized retention
target list — the kind of people-analytics deliverable that gets a BA a seat
in HR/People planning conversations.

## 3. Solution Overview
Synthetic employee and exit records in SQLite; SQL turnover-rate views by
department/level/tenure; a Python classification model scoring each active
employee's attrition risk; a configurable cost-per-attrition model; a
Streamlit dashboard.

## 4. Tech Stack
| Layer | Tool | Purpose |
|---|---|---|
| Language | Python 3.11+ | Core logic |
| Database | SQLite | Local relational store |
| Query layer | SQL | Turnover-rate views |
| Modeling | scikit-learn (LogisticRegression / RandomForestClassifier) | Attrition-risk scoring |
| Data handling | pandas, numpy | Transformation |
| App/dashboard | Streamlit | Interactive UI |
| Charts | Plotly | Turnover-by-department, risk distribution |
| Testing | pytest | Unit tests |
| Version control | Git + GitHub | Source control |

## 5. Architecture / Pipeline
```
[generate_synthetic_data.py] --> [SQLite: attritionradar.db]
        v
[queries/turnover_rates.sql] --> turnover % by dept/level/tenure bucket
        v
[attrition_model.py] -- feature engineering + classifier -->
        risk_score (0-1) per active employee
        v
[cost_model.py] --> $ cost exposure = at-risk headcount x cost-per-attrition
        v
[app.py: Streamlit] --> KPI cards, turnover chart, at-risk employee table
```
**Ingest (synthetic generator) → Store (SQLite) → Transform (SQL turnover
views) → Model (classifier + cost model) → Visualize (Streamlit) → Recommend
(at-risk table)**

## 6. Data Model
**`employees`**
| Column | Type |
|---|---|
| employee_id | INTEGER PK |
| department | TEXT |
| job_level | TEXT (IC1–IC5, M1–M3, etc.) |
| tenure_months | INTEGER |
| salary_band | TEXT |
| performance_rating | INTEGER (1–5) |
| last_promotion_months_ago | INTEGER |
| manager_id | INTEGER |

**`exits`**
| Column | Type |
|---|---|
| employee_id | INTEGER FK |
| exit_date | DATE |
| exit_reason | TEXT (voluntary/involuntary) |
| is_regretted | BOOLEAN |

## 7. Synthetic Data Generation Rules
- ~3,000 employees across 6–8 departments and 5–6 job levels
- Build voluntary-attrition probability as a weighted function of long
  `last_promotion_months_ago`, certain higher-turnover departments (e.g.,
  Sales, Support), and lower salary_band relative to job_level, then sample
  exits from it — so the later model has real, findable signal
- `is_regretted = True` more often when `performance_rating >= 4` (losing a
  strong performer)
- Split into ~80% still-active, ~20% exited over a simulated 24-month window

## 8. Modeling Details
- Features: tenure_months, last_promotion_months_ago, performance_rating,
  department (one-hot), job_level (one-hot), salary_band (one-hot)
- Model: start with `LogisticRegression`; also fit a
  `RandomForestClassifier` and log both AUCs so the README can state which
  performed better and why
- Output: `risk_score` (0–1) for every **active** employee, plus the top
  contributing feature per employee (from logistic-regression coefficients
  or random-forest feature importances)
- **Cost model (`cost_model.py`):** a lookup table of cost-per-attrition as a
  % of annual salary by job level (e.g., IC1–2: 50%, IC3–4: 80%, M1+: 120%),
  stored as named constants — never hardcoded inline — multiplied by
  at-risk headcount (employees above a risk threshold) to get $ exposure.
  State explicitly in the README that this is a **configurable assumption**,
  not a guarantee.

## 9. Dashboard Specification
- **Sidebar:** department filter, risk-score threshold slider
- **KPI row:** overall turnover rate, regretted-loss rate, # employees above
  risk threshold, estimated annual cost exposure ($)
- **Chart 1:** turnover rate by department (bar)
- **Chart 2:** risk-score distribution across active employees (histogram)
- **Table:** at-risk employees — employee_id, department, job_level,
  risk_score, top_risk_driver, sorted by risk_score descending

## 10. File & Folder Structure
```
attritionradar/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── generate_synthetic_data.py
├── tests/
│   ├── test_attrition_model.py
│   └── test_cost_model.py
├── queries/
│   └── turnover_rates.sql
└── src/
    ├── __init__.py
    ├── db.py
    ├── attrition_model.py
    └── cost_model.py
```

## 11. Step-by-Step Build Order
1. Scaffold folders; `.gitignore`, `LICENSE`.
2. Write `data/generate_synthetic_data.py` per Section 7; populates
   `attritionradar.db`.
3. Write `src/db.py`: connection + query helpers.
4. Write `queries/turnover_rates.sql`.
5. Write `src/attrition_model.py`: feature engineering + `train_and_score()
   -> DataFrame` per Section 8, logging both models' AUC.
6. Write `src/cost_model.py`: the cost-per-attrition lookup table and
   `estimate_cost_exposure(at_risk_df) -> float`, with every constant named
   and comment-documented.
7. Write `tests/`: assert risk_score is in [0,1] for all active employees;
   assert cost exposure scales linearly with headcount at a fixed risk
   threshold; assert changing the cost-per-attrition constants changes the
   output (i.e., it's actually being used, not hardcoded elsewhere).
8. Write `app.py` wiring db → attrition_model → cost_model → dashboard
   (Section 9).
9. Run locally, fix all exceptions.
10. Write final `README.md` per Section 14.
11. `git init`, commit, push.

## 12. Production-Quality Bar
- [ ] Type hints + docstrings
- [ ] No bare `except:`
- [ ] All cost assumptions are named constants in one place
  (`cost_model.py`), never magic numbers
- [ ] `logging` for model AUC and training row counts
- [ ] `pytest` passes
- [ ] `requirements.txt` pinned
- [ ] Zero unhandled exceptions

## 13. Roadmap / Future Enhancements
- Add a manager-level rollup (which managers have the highest regretted-loss
  rate)
- Try a gradient-boosted model (XGBoost/LightGBM) and compare AUC against
  the baseline two
- Recreate the at-risk table in Power BI/Tableau connected to the same
  SQLite file
- Add a "what-if" slider: simulate the effect of a salary-band adjustment on
  department-level risk

## 14. Required Contents of Final `README.md`
Business problem → schema summary → tech stack → setup steps → how
risk_score and cost exposure are computed (plain language) → an explicit
"cost-per-attrition is an adjustable assumption, not a guarantee" limitations
note → a synthetic-data disclosure (no real employee data used).

## 15. Definition of Done
- [ ] Both models train and score without error
- [ ] Cost exposure updates correctly when the risk threshold slider moves
- [ ] Tests pass
- [ ] README complete

## 16. Resume Bullet Template
"Built an HR attrition-prediction pipeline (SQL, Python/scikit-learn,
Streamlit) on synthetic workforce data, flagging [Y]% of at-risk employees
and $[X] in estimated attrition cost exposure."