import sqlite3
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

DB_PATH = Path("inventory.db")

def fetch_sales(weeks):
    placeholder = ",".join("?" for _ in weeks)
    qry = f"SELECT * FROM weekly_sales WHERE Week IN ({placeholder})"
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(qry, con, params=weeks)

def train_and_predict(week:int):
    # determine last up to 3 weeks
    start = max(1, week-3)
    train_weeks = list(range(start, week))
    if not train_weeks:
        raise ValueError("Not enough prior weeks to train on.")
    
    df_train = fetch_sales(train_weeks)
    df_test  = fetch_sales([week])
    if df_test.empty:
        raise ValueError(f"No data for week {week} to predict.")

    target   = "Quantity_Sold"
    features = [
        "Price_Bought", "Price_Sold", "Quantity_Bought",
        "Discount_Rate", "CAGR_Units_Sold", "Price_Elasticity",
        "Promo_Flag", "Stockout_Flag"
    ]
    cat_col  = ["Product_Name"]

    X_train = df_train[features + cat_col]
    y_train = df_train[target]
    X_test  = df_test[features + cat_col]

    # Preprocessing: impute numeric, then encode categoricals
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])
    preproc = ColumnTransformer([
        ("num", numeric_transformer, features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_col),
    ])

    # RandomForest pipeline
    rf = Pipeline([
        ("pre", preproc),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_train, y_train)
    df_test["RF_Pred"] = rf.predict(X_test).round().astype(int)

    # XGBoost pipeline
    xgb = Pipeline([
        ("pre", preproc),
        ("xgb", XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            objective="reg:squarederror",
            random_state=42
        ))
    ])
    xgb.fit(X_train, y_train)
    df_test["XGB_Pred"] = xgb.predict(X_test).round().astype(int)

    df_test["Predicted_Qty"] = ((df_test["RF_Pred"] + df_test["XGB_Pred"]) / 2).round().astype(int)
    return df_test[["Product_Name","Quantity_Bought","Predicted_Qty"]]
