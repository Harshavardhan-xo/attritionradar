# AttritionRadar

### Workforce Risk Command Center

**[Open Live Dashboard ↗](https://harsha-attritionradar.streamlit.app)**

Built around **15,000 employees** to demonstrate enterprise-style people analytics.

## Key capabilities

- Logistic Regression and Random Forest risk models
- Held-out AUC validation
- Department × job-level turnover benchmarking
- Active-employee risk explorer
- Regretted-loss and cost-exposure KPIs
- Promotion-lag sensitivity scenario
- Downloadable at-risk workforce list

## Cost model

Cost exposure uses configurable replacement-cost assumptions by job level. These are assumptions for portfolio demonstration, not guarantees.

## Data disclosure

Synthetic workforce data only. No sensitive personal attributes are used.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
