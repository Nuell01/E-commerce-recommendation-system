
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json

DATA_DIR = "/home/claude/project/data"
PBI_DIR = f"{DATA_DIR}/powerbi"

df = pd.read_pickle(f"{DATA_DIR}/clean_transactions.pkl")

SPLIT_DATE = pd.Timestamp("2011-11-01")
train = df[df["InvoiceDate"] < SPLIT_DATE].copy()
test = df[df["InvoiceDate"] >= SPLIT_DATE].copy()

train_customers = set(train["CustomerID"].unique())
test = test[test["CustomerID"].isin(train_customers)].copy()  
eval_customers = sorted(test["CustomerID"].unique())

print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
print(f"Train customers: {len(train_customers):,} | Customers evaluable (also in test): {len(eval_customers):,}")


products = sorted(train["StockCode"].unique())
customers = sorted(train["CustomerID"].unique())
prod_idx = {p: i for i, p in enumerate(products)}
cust_idx = {c: i for i, c in enumerate(customers)}

ui = train.groupby(["CustomerID", "StockCode"])["Quantity"].sum().reset_index()
rows = ui["CustomerID"].map(cust_idx)
cols = ui["StockCode"].map(prod_idx)
ui_mat = sparse.csr_matrix((ui["Quantity"].values, (rows, cols)), shape=(len(customers), len(products))) # this multiply customers x products

ui_mat_log = ui_mat.copy()
ui_mat_log.data = np.log1p(ui_mat_log.data)

purchased_train = train.groupby("CustomerID")["StockCode"].apply(set).to_dict()
purchased_test = test.groupby("CustomerID")["StockCode"].apply(set).to_dict()


pop_rank = train.groupby("StockCode")["CustomerID"].nunique().sort_values(ascending=False)
top_popular = list(pop_rank.index)

def recommend_popularity(customer_id, k=10):
    seen = purchased_train.get(customer_id, set())
    recs = [p for p in top_popular if p not in seen]
    return recs[:k]

 
item_sim = cosine_similarity(ui_mat_log.T, dense_output=False)  
item_sim = sparse.csr_matrix(item_sim)

def recommend_item_cf(customer_id, k=10):
    if customer_id not in cust_idx:
        return recommend_popularity(customer_id, k)
    u = cust_idx[customer_id]
    user_vec = ui_mat_log[u]  # 1 x n_products
    scores = user_vec.dot(item_sim).toarray().ravel() 
    seen = purchased_train.get(customer_id, set())
    seen_idx = {prod_idx[p] for p in seen if p in prod_idx}
    scores[list(seen_idx)] = -1  
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    top_idx = [i for i in top_idx if scores[i] > 0]
    recs = [products[i] for i in top_idx]
    if len(recs) < k:
        backfill = [p for p in top_popular if p not in seen and p not in recs]
        recs += backfill[: k - len(recs)]
    return recs[:k]

prod_desc = train.drop_duplicates("StockCode").set_index("StockCode")["Description"].reindex(products)
tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
tfidf_mat = tfidf.fit_transform(prod_desc.fillna(""))
content_sim = cosine_similarity(tfidf_mat, dense_output=False)
content_sim = sparse.csr_matrix(content_sim)

def recommend_content(customer_id, k=10):
    seen = purchased_train.get(customer_id, set())
    seen_idx = [prod_idx[p] for p in seen if p in prod_idx]
    if not seen_idx:
        return recommend_popularity(customer_id, k)
    scores = np.asarray(content_sim[seen_idx].sum(axis=0)).ravel()
    scores[seen_idx] = -1
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    top_idx = [i for i in top_idx if scores[i] > 0]
    recs = [products[i] for i in top_idx]
    if len(recs) < k:
        backfill = [p for p in top_popular if p not in seen and p not in recs]
        recs += backfill[: k - len(recs)]
    return recs[:k]


def recommend_hybrid(customer_id, k=10, w_cf=0.7, w_content=0.3):
    if customer_id not in cust_idx:
        return recommend_popularity(customer_id, k)
    u = cust_idx[customer_id]
    cf_scores = ui_mat_log[u].dot(item_sim).toarray().ravel()
    cf_max = cf_scores.max() if cf_scores.max() > 0 else 1.0
    cf_scores = cf_scores / cf_max

    seen = purchased_train.get(customer_id, set())
    seen_idx_list = [prod_idx[p] for p in seen if p in prod_idx]
    if seen_idx_list:
        content_scores = np.asarray(content_sim[seen_idx_list].sum(axis=0)).ravel()
        c_max = content_scores.max() if content_scores.max() > 0 else 1.0
        content_scores = content_scores / c_max
    else:
        content_scores = np.zeros(len(products))

    scores = w_cf * cf_scores + w_content * content_scores
    seen_idx = {prod_idx[p] for p in seen if p in prod_idx}
    scores[list(seen_idx)] = -1
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    top_idx = [i for i in top_idx if scores[i] > 0]
    recs = [products[i] for i in top_idx]
    if len(recs) < k:
        backfill = [p for p in top_popular if p not in seen and p not in recs]
        recs += backfill[: k - len(recs)]
    return recs[:k]


K = 10

def evaluate(rec_fn, name):
    precisions, recalls, hits = [], [], []
    for c in eval_customers:
        actual = purchased_test.get(c, set())
        if not actual:
            continue
        recs = set(rec_fn(c, K))
        n_correct = len(recs & actual)
        precisions.append(n_correct / K)
        recalls.append(n_correct / len(actual))
        hits.append(1 if n_correct > 0 else 0)
    return {
        "Model": name,
        "Precision@10": round(float(np.mean(precisions)), 4),
        "Recall@10": round(float(np.mean(recalls)), 4),
        "HitRate@10": round(float(np.mean(hits)), 4),
        "N_Customers_Evaluated": len(precisions),
    }

results = [
    evaluate(recommend_popularity, "Popularity Baseline"),
    evaluate(recommend_item_cf, "Item-Based Collaborative Filtering"),
    evaluate(recommend_content, "Content-Based (TF-IDF)"),
    evaluate(recommend_hybrid, "Hybrid (CF + Content)"),
]
results_df = pd.DataFrame(results).sort_values("Precision@10", ascending=False)
print("\n=== Evaluation Results (Top-10 recommendations) ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_fn = {
    "Popularity Baseline": recommend_popularity,
    "Item-Based Collaborative Filtering": recommend_item_cf,
    "Content-Based (TF-IDF)": recommend_content,
    "Hybrid (CF + Content)": recommend_hybrid,
}[best_model_name]
print(f"\nBest model: {best_model_name}")

results_df.to_csv(f"{PBI_DIR}/model_evaluation_results.csv", index=False)


trend_rows = []
test_months = sorted(test["InvoiceMonth"].unique())
for m in test_months:
    month_actual = test[test["InvoiceMonth"] == m].groupby("CustomerID")["StockCode"].apply(set).to_dict()
    precisions, recalls, hits = [], [], []
    for c, actual in month_actual.items():
        if c not in cust_idx:
            continue
        recs = set(best_fn(c, K))
        n_correct = len(recs & actual)
        precisions.append(n_correct / K)
        recalls.append(n_correct / len(actual))
        hits.append(1 if n_correct > 0 else 0)
    trend_rows.append({
        "Month": pd.Timestamp(m).strftime("%Y-%m-%d"),
        "Model": best_model_name,
        "Precision@10": round(float(np.mean(precisions)), 4) if precisions else None,
        "Recall@10": round(float(np.mean(recalls)), 4) if precisions else None,
        "HitRate@10": round(float(np.mean(hits)), 4) if precisions else None,
        "N_Customers": len(precisions),
    })
trend_df = pd.DataFrame(trend_rows)
trend_df.to_csv(f"{PBI_DIR}/accuracy_over_time.csv", index=False)
print("\nAccuracy over time (best model):")
print(trend_df.to_string(index=False))


full_products = sorted(df["StockCode"].unique())
full_customers = sorted(df["CustomerID"].unique())
fprod_idx = {p: i for i, p in enumerate(full_products)}
fcust_idx = {c: i for i, c in enumerate(full_customers)}
full_ui = df.groupby(["CustomerID", "StockCode"])["Quantity"].sum().reset_index()
frows = full_ui["CustomerID"].map(fcust_idx)
fcols = full_ui["StockCode"].map(fprod_idx)
full_mat = sparse.csr_matrix((full_ui["Quantity"].values, (frows, fcols)), shape=(len(full_customers), len(full_products)))
full_mat_log = full_mat.copy()
full_mat_log.data = np.log1p(full_mat_log.data)
full_item_sim = sparse.csr_matrix(cosine_similarity(full_mat_log.T, dense_output=False))
full_purchased = df.groupby("CustomerID")["StockCode"].apply(set).to_dict()
full_pop = df.groupby("StockCode")["CustomerID"].nunique().sort_values(ascending=False)
full_top_popular = list(full_pop.index)
full_desc_map = df.drop_duplicates("StockCode").set_index("StockCode")["Description"]

def final_recommend_item_cf(customer_id, k=10):
    u = fcust_idx[customer_id]
    scores = full_mat_log[u].dot(full_item_sim).toarray().ravel()
    seen = full_purchased.get(customer_id, set())
    seen_idx = {fprod_idx[p] for p in seen if p in fprod_idx}
    scores[list(seen_idx)] = -1
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    top_idx = [i for i in top_idx if scores[i] > 0]
    recs = [full_products[i] for i in top_idx]
    if len(recs) < k:
        backfill = [p for p in full_top_popular if p not in seen and p not in recs]
        recs += backfill[: k - len(recs)]
    return recs[:k]

final_rows = []
for c in full_customers:
    recs = final_recommend_item_cf(c, K)
    for rank, p in enumerate(recs, start=1):
        final_rows.append({
            "CustomerID": c,
            "Rank": rank,
            "Recommended_StockCode": p,
            "Recommended_Product": full_desc_map.get(p, ""),
        })
final_recs_df = pd.DataFrame(final_rows)
final_recs_df.to_csv(f"{PBI_DIR}/customer_recommendations_top10.csv", index=False)
print(f"\nGenerated {len(final_recs_df):,} recommendation rows for {final_recs_df['CustomerID'].nunique():,} customers.")

with open(f"{DATA_DIR}/model_summary.json", "w") as f:
    json.dump({
        "split_date": str(SPLIT_DATE.date()),
        "train_rows": len(train),
        "test_rows": len(test),
        "evaluation_results": results,
        "best_model": best_model_name,
    }, f, indent=2)
