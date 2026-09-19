import numpy as np, pandas as pd, streamlit as st, plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="AttritionRadar",layout="wide")
st.title("AttritionRadar — Employee Attrition & Workforce Cost")
st.caption("Synthetic people-analytics dashboard. Cost exposure is a configurable assumption.")

@st.cache_data
def make():
    rng=np.random.default_rng(42); n=3200
    dept=rng.choice(["Sales","Support","Engineering","Finance","HR","Operations","Marketing","Product"],n)
    level=rng.choice(["IC1","IC2","IC3","IC4","IC5","M1","M2","M3"],n)
    tenure=rng.integers(1,121,n); promo=np.minimum(tenure,rng.integers(0,49,n))
    perf=rng.integers(1,6,n); salary=rng.choice(["L1","L2","L3","L4","L5","L6","M4","M5","M6","M7"],n)
    z=-3+.045*promo+.65*np.isin(dept,["Sales","Support"])+.25*(perf<=2)-.35*(perf>=4)+rng.normal(0,.35,n)
    p=1/(1+np.exp(-z)); exit_flag=rng.random(n)<p; voluntary=exit_flag&(rng.random(n)>.15); regretted=voluntary&(perf>=4)
    return pd.DataFrame({"employee_id":np.arange(1,n+1),"department":dept,"job_level":level,"tenure_months":tenure,
                         "salary_band":salary,"performance_rating":perf,"last_promotion_months_ago":promo,
                         "exited":exit_flag,"voluntary_exit":voluntary,"is_regretted":regretted})

@st.cache_resource
def score(df):
    num=["tenure_months","last_promotion_months_ago","performance_rating"]; cat=["department","job_level","salary_band"]
    X=df[num+cat]; y=df.voluntary_exit
    a,b,ya,yb=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    pre=ColumnTransformer([("n",StandardScaler(),num),("c",OneHotEncoder(handle_unknown="ignore"),cat)])
    lr=Pipeline([("pre",pre),("m",LogisticRegression(max_iter=2000,random_state=42))])
    rf=Pipeline([("pre",pre),("m",RandomForestClassifier(n_estimators=220,class_weight="balanced",random_state=42))])
    lr.fit(a,ya); rf.fit(a,ya)
    auc_lr=roc_auc_score(yb,lr.predict_proba(b)[:,1]); auc_rf=roc_auc_score(yb,rf.predict_proba(b)[:,1])
    active=df[~df.exited].copy(); active["risk_score"]=rf.predict_proba(active[num+cat])[:,1]
    return active,float(auc_lr),float(auc_rf)

df=make(); active,auc_lr,auc_rf=score(df)
threshold=st.sidebar.slider("Risk threshold",0.0,1.0,.5,.05)
dept=st.sidebar.multiselect("Department",sorted(active.department.unique()),default=sorted(active.department.unique()))
risk=active[(active.risk_score>=threshold)&active.department.isin(dept)]

rate={"IC1":.5,"IC2":.5,"IC3":.8,"IC4":.8,"IC5":.8,"M1":1.2,"M2":1.2,"M3":1.2}
salary={"L1":450000,"L2":600000,"L3":800000,"L4":1000000,"L5":1300000,"L6":1650000,"M4":1600000,"M5":2000000,"M6":2500000,"M7":3200000}
cost=float((risk.job_level.map(rate).fillna(.8)*risk.salary_band.map(salary).fillna(800000)).sum())

c1,c2,c3,c4=st.columns(4)
c1.metric("Voluntary Turnover",f"{df.voluntary_exit.mean():.1%}")
c2.metric("Regretted Loss",f"{df.is_regretted.mean():.1%}")
c3.metric("Employees at Risk",f"{len(risk):,}")
c4.metric("Cost Exposure",f"INR {cost:,.0f}")

left,right=st.columns(2)
with left: st.plotly_chart(px.bar(df.groupby("department",as_index=False).voluntary_exit.mean(),x="department",y="voluntary_exit",title="Voluntary Turnover by Department"),use_container_width=True)
with right: st.plotly_chart(px.histogram(active,x="risk_score",nbins=30,title="Attrition Risk Distribution"),use_container_width=True)
st.dataframe(risk[["employee_id","department","job_level","risk_score","salary_band","performance_rating"]].sort_values("risk_score",ascending=False),use_container_width=True)
st.caption(f"Logistic AUC {auc_lr:.3f} | Random Forest AUC {auc_rf:.3f}")
