
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

DATA_DIR = "/home/claude/project/data"
PBI_DIR = f"{DATA_DIR}/powerbi"
import os
os.makedirs(PBI_DIR, exist_ok=True)

df = pd.read_pickle(f"{DATA_DIR}/clean_transactions.pkl")


fact = df[["InvoiceNo", "CustomerID", "StockCode", "Description", "Quantity",
           "UnitPrice", "LineTotal", "InvoiceDate", "InvoiceMonth", "Country"]].copy()
fact.to_csv(f"{PBI_DIR}/fact_transactions.csv", index=False)


snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency_Days=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency_Orders=("InvoiceNo", "nunique"),
    Monetary_Total=("LineTotal", "sum"),
    Total_Items=("Quantity", "sum"),
    Distinct_Products=("StockCode", "nunique"),
    First_Purchase=("InvoiceDate", "min"),
    Last_Purchase=("InvoiceDate", "max"),
    Country=("Country", lambda x: x.mode().iat[0]),
).reset_index()

rfm["Avg_Order_Value"] = (rfm["Monetary_Total"] / rfm["Frequency_Orders"]).round(2)
rfm["Tenure_Days"] = (rfm["Last_Purchase"] - rfm["First_Purchase"]).dt.days


rfm["R_Score"] = pd.qcut(rfm["Recency_Days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency_Orders"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary_Total"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

def segment(row):
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal Customers"
    if r >= 4 and f <= 2:
        return "New / Promising"
    if r <= 2 and f >= 4:
        return "At Risk (used to buy a lot)"
    if r <= 2 and f <= 2 and m <= 2:
        return "Hibernating / Lost"
    return "Needs Attention"

rfm["Segment"] = rfm.apply(segment, axis=1)
rfm.round(2).to_csv(f"{PBI_DIR}/dim_customers_rfm.csv", index=False)


prod = df.groupby(["StockCode", "Description"]).agg(
    Total_Qty_Sold=("Quantity", "sum"),
    Total_Revenue=("LineTotal", "sum"),
    N_Orders=("InvoiceNo", "nunique"),
    N_Customers=("CustomerID", "nunique"),
    Avg_Price=("UnitPrice", "mean"),
).reset_index()
prod["Revenue_Rank"] = prod["Total_Revenue"].rank(ascending=False, method="min").astype(int)
prod["Qty_Rank"] = prod["Total_Qty_Sold"].rank(ascending=False, method="min").astype(int)
prod = prod.sort_values("Revenue_Rank")
prod.round(2).to_csv(f"{PBI_DIR}/dim_products.csv", index=False)


monthly = df.groupby("InvoiceMonth").agg(
    Revenue=("LineTotal", "sum"),
    Orders=("InvoiceNo", "nunique"),
    Customers=("CustomerID", "nunique"),
    Items=("Quantity", "sum"),
).reset_index()
monthly["Avg_Order_Value"] = (monthly["Revenue"] / monthly["Orders"]).round(2)
monthly.round(2).to_csv(f"{PBI_DIR}/monthly_sales.csv", index=False)

# day-of-week purchase timing
dow_hour = df.groupby(["InvoiceDow", "InvoiceHour"]).agg(
    Revenue=("LineTotal", "sum"),
    Orders=("InvoiceNo", "nunique"),
).reset_index()
dow_hour.round(2).to_csv(f"{PBI_DIR}/dow_hour_sales.csv", index=False)

#country
country = df.groupby("Country").agg(
    Revenue=("LineTotal", "sum"),
    Orders=("InvoiceNo", "nunique"),
    Customers=("CustomerID", "nunique"),
).reset_index().sort_values("Revenue", ascending=False)
country.round(2).to_csv(f"{PBI_DIR}/country_sales.csv", index=False)


basket = df.groupby("InvoiceNo")["StockCode"].apply(set)
basket = basket[basket.apply(len) > 1]
pair_counter = Counter()
for items in basket:
    items = sorted(items)
    if len(items) > 30:  
        continue
    for a, b in combinations(items, 2):
        pair_counter[(a, b)] += 1

top_pairs = pair_counter.most_common(500)
desc_map = df.drop_duplicates("StockCode").set_index("StockCode")["Description"]
pairs_df = pd.DataFrame(
    [(a, desc_map.get(a, ""), b, desc_map.get(b, ""), c) for (a, b), c in top_pairs],
    columns=["Product_A_Code", "Product_A_Name", "Product_B_Code", "Product_B_Name", "Times_Bought_Together"],
)
pairs_df.to_csv(f"{PBI_DIR}/cross_sell_pairs.csv", index=False)

print("RFM segment distribution:")
print(rfm["Segment"].value_counts())
print("\nTop 10 products by revenue:")
print(prod.head(10)[["Description", "Total_Revenue", "Total_Qty_Sold"]].to_string(index=False))
print("\nTop 10 cross-sell pairs:")
print(pairs_df.head(10)[["Product_A_Name", "Product_B_Name", "Times_Bought_Together"]].to_string(index=False))
print("\nMonthly sales:")
print(monthly[["InvoiceMonth", "Revenue", "Orders", "Customers"]].to_string(index=False))
