from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚘", layout="wide")
DATA_CANDIDATES = [Path("car data.csv"), Path("CAR DETAILS FROM CAR DEKHO.csv"), Path("car_data.csv"), Path("outputs/car data.csv")]

def load_data(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    for p in DATA_CANDIDATES:
        if p.exists(): return pd.read_csv(p)
    return None

def normalize(df):
    df=df.copy(); df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]
    aliases={"selling_price":"selling_price", "price":"selling_price", "present_price":"present_price", "year":"year", "kms_driven":"km_driven", "km_driven":"km_driven", "fuel_type":"fuel", "fuel":"fuel", "seller_type":"seller_type", "transmission":"transmission", "owner":"owner", "car_name":"name", "name":"name"}
    df=df.rename(columns={c:aliases[c] for c in df.columns if c in aliases})
    if "selling_price" not in df: raise ValueError("Dataset must include Selling_Price (or Price).")
    if "year" in df: df["vehicle_age"]=pd.Timestamp.now().year-pd.to_numeric(df["year"],errors="coerce")
    elif "vehicle_age" not in df: df["vehicle_age"]=None
    for col in ["present_price","km_driven","owner"]:
        if col not in df: df[col]=None
    for col in ["fuel","seller_type","transmission","name"]:
        if col not in df: df[col]="Unknown"
    for col in ["selling_price","present_price","km_driven","owner","vehicle_age"]:
        df[col]=pd.to_numeric(df[col],errors="coerce")
    df=df.dropna(subset=["selling_price"]); df=df[df.selling_price>0]
    return df

def fit_model(df):
    features=["vehicle_age","km_driven","present_price","owner","fuel","seller_type","transmission","name"]
    X=df[features]; y=df["selling_price"]
    numeric=["vehicle_age","km_driven","present_price","owner"]
    categorical=["fuel","seller_type","transmission","name"]
    prep=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median", add_indicator=True)),("scale",StandardScaler())]),numeric),
      ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
    models={"Random Forest":RandomForestRegressor(n_estimators=300,min_samples_leaf=1,max_features=0.9,random_state=42),"Extra Trees":ExtraTreesRegressor(n_estimators=300,min_samples_leaf=1,max_features=0.9,random_state=42),"Gradient Boosting":GradientBoostingRegressor(n_estimators=150,learning_rate=0.04,max_depth=2,random_state=42)}
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    rows=[]; fitted={}
    for name,reg in models.items():
        pipe=Pipeline([("prep",prep),("model",reg)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        rows.append({"Model":name,"MAE":mean_absolute_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**0.5,"R²":r2_score(yte,pred)})
        fitted[name]=pipe
    results=pd.DataFrame(rows).sort_values("RMSE"); best=results.iloc[0]["Model"]
    final=Pipeline([("prep",prep),("model",models[best])]); final.fit(X,y)
    return final,results,best,features

st.title("🚘 Used Car Price Prediction")
st.write("Estimate a resale price from vehicle age, mileage, fuel, and other listing details.")
st.info("Use the CarDekho dataset from Kaggle. Upload its CSV below, or put `car data.csv` beside app.py. Download link: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho")
upload=st.file_uploader("Upload CarDekho CSV",type="csv")
try:
    raw=load_data(upload)
    if raw is None: st.warning("Upload the CarDekho CSV to begin."); st.stop()
    data=normalize(raw)
    if len(data)<10: st.error("At least 10 valid rows are needed."); st.stop()
    model,metrics,best,features=fit_model(data)
except Exception as e:
    st.error(f"Could not prepare the dataset: {e}"); st.stop()
with st.expander("Dataset and model results", expanded=True):
    a,b,c=st.columns(3); a.metric("Usable listings",f"{len(data):,}"); b.metric("Selected model",best); c.metric("Holdout RMSE",f"{metrics.iloc[0]['RMSE']:,.2f}")
    st.dataframe(metrics.style.format({"MAE":"{:,.2f}","RMSE":"{:,.2f}","R²":"{:.3f}"}),use_container_width=True)
st.subheader("Enter vehicle details")
left,right=st.columns(2)
with left:
    age=st.number_input("Vehicle age (years)",0,40,5); km=st.number_input("Kilometres driven",0,1000000,50000,step=1000)
    present=st.number_input("Original/current showroom price (dataset units)",0.0,10000000.0,5.0,step=0.5)
    owner=st.number_input("Previous owners",0,10,0)
with right:
    fuel=st.selectbox("Fuel type",sorted(data.fuel.dropna().astype(str).unique()))
    seller=st.selectbox("Seller type",sorted(data.seller_type.dropna().astype(str).unique()))
    trans=st.selectbox("Transmission",sorted(data.transmission.dropna().astype(str).unique()))
    name=st.selectbox("Car/model",sorted(data.name.dropna().astype(str).unique()))
if st.button("Estimate resale price",type="primary"):
    row=pd.DataFrame([{"vehicle_age":age,"km_driven":km,"present_price":present,"owner":owner,"fuel":fuel,"seller_type":seller,"transmission":trans,"name":name}])
    price=max(0,float(model.predict(row[features])[0])); st.success(f"Estimated resale price: {price:,.2f} (same units as dataset Selling_Price)")
    st.caption("This is a data-based estimate, not a valuation guarantee. Dataset prices are commonly recorded in lakh rupees; check the Kaggle file's units.")
