
import pandas as pd
import numpy as np

RAW_PATH = "/mnt/user-data/uploads/Online_Retail.xlsx"
OUT_DIR = "/home/claude/project/data"

df = pd.read_excel(RAW_PATH)
n_raw = len(df)


df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
df["Description"] = df["Description"].astype(str).str.strip()
df["Country"] = df["Country"].astype(str).str.strip()
df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()

report = {"rows_raw": n_raw}


missing_cust = df["CustomerID"].isna().sum()
df = df[df["CustomerID"].notna()].copy()
df["CustomerID"] = df["CustomerID"].astype(int)
report["dropped_missing_customer_id"] = int(missing_cust)


is_cancel = df["InvoiceNo"].str.startswith("C")
report["dropped_cancelled_invoices"] = int(is_cancel.sum())
df = df[~is_cancel].copy()

bad_qty = df["Quantity"] <= 0
bad_price = df["UnitPrice"] <= 0
report["dropped_non_positive_qty"] = int(bad_qty.sum())
report["dropped_non_positive_price"] = int((bad_price & ~bad_qty).sum())
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()


junk_codes = {"POST", "D", "DOT", "M", "MANUAL", "BANK CHARGES", "PADS", "CRUK", "C2", "AMAZONFEE"}
is_junk = df["StockCode"].isin(junk_codes)
report["dropped_admin_stockcodes"] = int(is_junk.sum())
df = df[~is_junk].copy()


n_before = len(df)
df = df.drop_duplicates()
report["dropped_exact_duplicates"] = int(n_before - len(df))


desc_map = df.dropna(subset=["Description"]).groupby("StockCode")["Description"].agg(
    lambda x: x.mode().iat[0] if not x.mode().empty else np.nan
)
df["Description"] = df["Description"].replace("nan", np.nan)
df["Description"] = df["Description"].fillna(df["StockCode"].map(desc_map))
n_before = len(df)
df = df.dropna(subset=["Description"])
report["dropped_unresolvable_description"] = int(n_before - len(df))


q_cap = df["Quantity"].quantile(0.995)
p_cap = df["UnitPrice"].quantile(0.995)
report["quantity_cap_995pct"] = float(q_cap)
report["price_cap_995pct"] = float(p_cap)
df["Quantity"] = df["Quantity"].clip(upper=q_cap)
df["UnitPrice"] = df["UnitPrice"].clip(upper=p_cap)

# --- derived fields ---
df["LineTotal"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["InvoiceMonth"] = df["InvoiceDate"].values.astype("datetime64[M]")
df["InvoiceYear"] = df["InvoiceDate"].dt.year
df["InvoiceDow"] = df["InvoiceDate"].dt.day_name()
df["InvoiceHour"] = df["InvoiceDate"].dt.hour

report["rows_clean"] = int(len(df))
report["pct_retained"] = round(100 * len(df) / n_raw, 1)
report["n_customers"] = int(df["CustomerID"].nunique())
report["n_products"] = int(df["StockCode"].nunique())
report["n_invoices"] = int(df["InvoiceNo"].nunique())
report["date_min"] = str(df["InvoiceDate"].min())
report["date_max"] = str(df["InvoiceDate"].max())
report["n_countries"] = int(df["Country"].nunique())

df.to_pickle(f"{OUT_DIR}/clean_transactions.pkl")
df.to_csv(f"{OUT_DIR}/clean_transactions_full.csv", index=False)
df.head(2000).to_csv(f"{OUT_DIR}/clean_transactions_sample.csv", index=False)

import json
with open(f"{OUT_DIR}/cleaning_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
