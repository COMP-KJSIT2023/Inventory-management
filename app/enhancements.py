import pandas as pd

def enhance_weekly(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame `df_all` with columns:
      ['Product_Name','Price_Bought','Quantity_Bought','Price_Sold',
       'Quantity_Sold','Start_Date','End_Date','Week']
    returns a new DataFrame with these extra columns:
      - Discount_Rate
      - Promo_Flag
      - Stockout_Flag
      - Qty_Base       (only for week>1)
      - Week_Index     (only for week>1)
      - CAGR_Units_Sold (NaN for week1, else computed)
      - Lag_Qty, Lag_Price
      - Price_Elasticity (NaN for week1)
    """
    promo_cut = 0.10

    df = df_all.copy()
    df["Discount_Rate"]   = (df["Price_Sold"] - df["Price_Bought"]) / df["Price_Bought"]
    df["Promo_Flag"]      = (df["Discount_Rate"] >= promo_cut).astype(int)
    df["Stockout_Flag"]   = (df["Quantity_Sold"] == df["Quantity_Bought"]).astype(int)

    # sort + group
    df = df.sort_values(["Product_Name", "Week"])
    g  = df.groupby("Product_Name")

    # first‐week baseline & index
    df["Qty_Base"]     = g["Quantity_Sold"].transform("first")
    df["Week_Index"]   = g.cumcount() + 1

    # CAGR (only meaningful for Week_Index>1)
    df["CAGR_Units_Sold"] = (
        (df["Quantity_Sold"] / df["Qty_Base"])
        .pow(1 / (df["Week_Index"] - 1).replace(0, 1))
        - 1
    )
    df.loc[df["Week_Index"] == 1, "CAGR_Units_Sold"] = pd.NA

    # lag features
    df["Lag_Qty"]   = g["Quantity_Sold"].shift(1)
    df["Lag_Price"] = g["Price_Sold"].shift(1)

    # price elasticity (only for Week_Index>1)
    pct_qty   = (df["Quantity_Sold"] - df["Lag_Qty"])  / df["Lag_Qty"].replace(0, 1)
    pct_price = (df["Price_Sold"]   - df["Lag_Price"]) / df["Lag_Price"].replace(0, 1)
    df["Price_Elasticity"] = pct_qty / pct_price.replace(0, 1)
    df.loc[df["Week_Index"] == 1, "Price_Elasticity"] = pd.NA

    return df
