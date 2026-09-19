import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="AttritionRadar | Workforce Risk", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{padding:1.2rem 1.4rem;border-radius:18px;background:linear-gradient(135deg,#29131a,#61203a);color:white;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;color:#f9d5df}
.badge{display:inline-block;background:#e11d48;color:white;padding:.25rem .6rem;border-radius:999px;font-size:.75rem;font-weight:700}
</style>
""",unsafe_allow_html=True)

SALARY={"L1":450000,"L2":600000,"L3":800000,"L4":1000000,"L5":1300000,"L6":1650000,"M4":1600000,"M5":2000000,"M6":2500000,"M7":3200000}
COST_RATE={"IC1":.5,"IC2":.5,"IC3":.8,"IC4":.8,"IC5":.8,"M1":1.2,"M2":1.2,"M3":1.2}

@st.cache_data
def generate():
    rng=np.random.default_rng(24); n=15000
    departments=["Sales","Support","Engineering","Finance","HR","Operations","Marketing","Product"]
    levels=["IC1","IC2","IC3","IC4","IC5","M1","M2","M3"]; salary_bands=["L1","L2","L3","L4","L5","L6","M4","M5","M6","M7"]
    dept=rng.choice(departments,n); level=rng.choice(levels,n,p=[.16,.16,.18,.16,.11,.09,.07,.07])
    tenure=rng.integers(1,145,n); promo=np.minimum(tenure,rng.integers(0,61,n)); perf=np.clip(rng.normal(3.4,.8,n),1,5).round(0)
    salary=rng.choice(salary_bands,n); manager_span=rng.integers(2,15,n)
    engagement=np.clip(rng.normal(68-.25*promo+.5*perf,13,n),5,98)
    z=-3.2+.038*promo+.55*np.isin(dept,["Sales","Support"])+.18*(perf<=2)-.3*(perf>=4)-.012*engagement+.025*manager_span+rng.normal(0,.45,n)
    p=1/(1+np.exp(-z)); exited=rng.random(n)<p; voluntary=exited&(rng.random(n)>.15); regretted=voluntary&(perf>=4)
    return pd.DataFrame({"employee_id":np.arange(1,n+1),"department":dept,"job_level":level,"tenure_months":tenure,
                         "salary_band":salary,"performance_rating":perf.astype(int),"last_promotion_months_ago":promo,
                         "manager_span":manager_span,"engagement_score":engagement.round(1),"exited":exited,
                         "voluntary_exit":voluntary,"is_regretted":regretted})

@st.cache_resource
def score(df):
    num=["tenure_months","last_promotion_months_ago","performance_rating","manager_span","engagement_score"]; cat=["department","job_level","salary_band"]
    X=df[num+cat]; y=df.voluntary_exit.astype(int)
    a,b,ya,yb=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    pre=ColumnTransformer([("n",StandardScaler(),num),("c",OneHotEncoder(handle_unknown="ignore"),cat)])
    lr=Pipeline([("pre",pre),("m",LogisticRegression(max_iter=2500,class_weight="balanced",random_state=42))])
    rf=Pipeline([("pre",pre),("m",RandomForestClassifier(n_estimators=260,min_samples_leaf=4,class_weight="balanced_subsample",random_state=42,n_jobs=-1))])
    lr.fit(a,ya); rf.fit(a,ya)
    auc_lr=float(roc_auc_score(yb,lr.predict_proba(b)[:,1])); auc_rf=float(roc_auc_score(yb,rf.predict_proba(b)[:,1]))
    active=df[~df.exited].copy(); active["risk_score"]=rf.predict_proba(active[num+cat])[:,1]
    return active,auc_lr,auc_rf

df=generate(); active,auc_lr,auc_rf=score(df)
with st.sidebar:
    threshold=st.slider("Risk threshold",.20,.90,.55,.05)
    departments=st.multiselect("Department",sorted(active.department.unique()),default=sorted(active.department.unique()))
    levels=st.multiselect("Job level",sorted(active.job_level.unique()),default=sorted(active.job_level.unique()))
risk=active[(active.risk_score>=threshold)&active.department.isin(departments)&active.job_level.isin(levels)].copy()
risk["annual_salary"]=risk.salary_band.map(SALARY); risk["cost_exposure"]=risk.job_level.map(COST_RATE)*risk.annual_salary

st.markdown('<div class="hero"><span class="badge">PEOPLE ANALYTICS • WORKFORCE PLANNING</span><h1>AttritionRadar — Workforce Risk Command Center</h1><p>Segment-level turnover, active-employee risk scoring, regretted-loss monitoring and configurable cost exposure across a 15,000-employee synthetic workforce.</p></div>',unsafe_allow_html=True)
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Headcount",f"{len(df):,}"); c2.metric("Voluntary Turnover",f"{df.voluntary_exit.mean():.1%}")
c3.metric("Regretted Loss",f"{df.is_regretted.mean():.1%}"); c4.metric("Employees at Risk",f"{len(risk):,}")
c5.metric("Cost Exposure",f"₹{risk.cost_exposure.sum()/1e7:.2f}Cr")
st.caption(f"Model validation: Logistic AUC {auc_lr:.3f} • Random Forest AUC {auc_rf:.3f} • Data scale: {len(df):,} employees")

t1,t2,t3,t4=st.tabs(["Executive Overview","Risk Explorer","Department Benchmark","Scenario Planner"])
with t1:
    by=df.groupby("department",as_index=False).agg(headcount=("employee_id","count"),turnover=("voluntary_exit","mean"),regretted=("is_regretted","mean"))
    l,r=st.columns(2)
    l.plotly_chart(px.bar(by,x="department",y="turnover",title="Voluntary Turnover by Department"),use_container_width=True)
    r.plotly_chart(px.bar(df.groupby("job_level",as_index=False).voluntary_exit.mean(),x="job_level",y="voluntary_exit",title="Turnover by Job Level"),use_container_width=True)
    st.dataframe(by.sort_values("turnover",ascending=False),use_container_width=True,hide_index=True)
with t2:
    l,r=st.columns(2); sample=active.sample(min(8000,len(active)),random_state=42)
    l.plotly_chart(px.scatter(sample,x="engagement_score",y="risk_score",color="job_level",hover_data=["employee_id","department","tenure_months","performance_rating"],title="Engagement vs Predicted Risk"),use_container_width=True)
    r.plotly_chart(px.histogram(active,x="risk_score",color="department",nbins=30,title="Risk Distribution"),use_container_width=True)
    cols=["employee_id","department","job_level","tenure_months","salary_band","performance_rating","engagement_score","risk_score"]
    st.dataframe(risk[cols].sort_values("risk_score",ascending=False),use_container_width=True,hide_index=True)
    st.download_button("Download At-Risk Workforce List",risk[cols].to_csv(index=False).encode(),"attritionradar_at_risk.csv","text/csv")
with t3:
    heat=df.pivot_table(index="department",columns="job_level",values="voluntary_exit",aggfunc="mean")
    st.plotly_chart(px.imshow(heat,text_auto=".1%",aspect="auto",color_continuous_scale="Reds",title="Turnover Heatmap: Department × Level"),use_container_width=True)
    ten=df.assign(tenure_band=pd.cut(df.tenure_months,[0,12,24,48,84,999],labels=["0–1y","1–2y","2–4y","4–7y","7y+"]))
    st.plotly_chart(px.bar(ten.groupby("tenure_band",as_index=False,observed=True).voluntary_exit.mean(),x="tenure_band",y="voluntary_exit",title="Turnover by Tenure Band"),use_container_width=True)
with t4:
    promo_cap=st.slider("Scenario: maximum months since promotion",6,48,24,3)
    base=active.risk_score.mean()
    scenario=active.risk_score*np.where(active.last_promotion_months_ago>promo_cap,.82,1.0)
    delta=max(0,base-scenario.mean()); current_cost=(active.job_level.map(COST_RATE)*active.salary_band.map(SALARY)).sum(); scen_cost=current_cost*(1-delta*.8)
    a,b,c=st.columns(3); a.metric("Current Avg Risk",f"{base:.1%}"); b.metric("Scenario Avg Risk",f"{scenario.mean():.1%}",f"-{delta:.1%}"); c.metric("Illustrative Cost",f"₹{scen_cost/1e7:.2f}Cr")
    st.info("Sensitivity analysis only: the scenario applies a transparent adjustment to modeled risk; it is not a causal forecast.")
