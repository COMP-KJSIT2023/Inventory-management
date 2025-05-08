import pandas as pd
import numpy as np

df = pd.read_csv('./dmart_products.csv')
df2 = pd.DataFrame()
# Fixing dtype issues by ensuring numeric conversion
df2["Product_Name"] = df["Product Name"].astype(str)
df2["Price_Bought"] = df["Price (₹)"]
df2["Quantity_Bought"] = np.random.randint(200, 600, size=len(df2))

# Recalculate derived columns
df2["Price_Sold"] = df2["Price_Bought"] * np.random.uniform(0.85, 1.05, size=len(df))
df2["Price_Sold"] = df2["Price_Sold"].round(2)

df2["Quantity_Sold"] = (df2["Quantity_Bought"] * np.random.uniform(0.7, 1.0, size=len(df))).astype(int)

# Add fixed date range
df2["Start_Date"] = "22-03-2025"
df2["End_Date"] = "28-03-2025"

# Reorder
df2 = df2[[
    "Product_Name", "Price_Bought", "Quantity_Bought",
    "Price_Sold", "Quantity_Sold", "Start_Date", "End_Date"
]]

df2.to_csv("./dataset/retail_week4.csv", index=False)
