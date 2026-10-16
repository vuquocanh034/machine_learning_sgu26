import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA = "/home/claude/hp/data"
FIGS = "/home/claude/hp/figs"
OUT = "/home/claude/hp/outputs"
os.makedirs(FIGS, exist_ok=True); os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
results = {"train_shape": train.shape, "test_shape": test.shape}

plt.figure(figsize=(8,5))
sns.histplot(train["SalePrice"], kde=True, color="teal")
plt.title("Phân phối SalePrice"); plt.tight_layout()
plt.savefig(f"{FIGS}/01_saleprice_dist.png", dpi=130); plt.close()

plt.figure(figsize=(10,6))
sns.heatmap(train.isnull(), cbar=False, cmap="viridis")
plt.title("Bản đồ giá trị thiếu - Train"); plt.tight_layout()
plt.savefig(f"{FIGS}/02_missing_train.png", dpi=130); plt.close()

num_train = train.select_dtypes(include=[np.number])
corr = num_train.corr()
top_corr_cols = corr["SalePrice"].abs().sort_values(ascending=False).head(15).index
plt.figure(figsize=(12,10))
sns.heatmap(train[top_corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Ma trận tương quan - Top 15 biến"); plt.tight_layout()
plt.savefig(f"{FIGS}/03_corr_heatmap.png", dpi=130); plt.close()
results["top_corr"] = corr["SalePrice"].abs().sort_values(ascending=False).head(11).to_dict()

cols_to_drop = ["Alley", "PoolQC", "Fence", "MiscFeature", "Id"]
y = train["SalePrice"].copy()
test_ids = test["Id"].copy()
train_feat = train.drop(columns=cols_to_drop + ["SalePrice"], errors="ignore")
test_feat = test.drop(columns=cols_to_drop, errors="ignore")

def fill_missing(df):
    df = df.copy()
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isnull().any(): df[c] = df[c].fillna(df[c].mean())
    for c in df.select_dtypes(exclude=[np.number]).columns:
        if df[c].isnull().any(): df[c] = df[c].fillna(df[c].mode()[0])
    return df

train_feat = fill_missing(train_feat)
test_feat = fill_missing(test_feat)

n_train = train_feat.shape[0]
combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
cat_cols = combined.select_dtypes(exclude=[np.number]).columns.tolist()
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
X_train_full = combined_encoded.iloc[:n_train, :].reset_index(drop=True)
X_test_final = combined_encoded.iloc[n_train:, :].reset_index(drop=True)
results["n_features"] = X_train_full.shape[1]
results["X_train_shape"] = X_train_full.shape
results["X_test_shape"] = X_test_final.shape

X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
rf_pred = rf.predict(X_val)
rf_rmse = float(np.sqrt(mean_squared_error(y_val, rf_pred)))
rf_mae = float(mean_absolute_error(y_val, rf_pred))
rf_r2 = float(r2_score(y_val, rf_pred))
results["random_forest"] = {"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2}

importances = pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(9,7))
sns.barplot(x=importances.values, y=importances.index, color="steelblue")
plt.title("Random Forest - Top 15 đặc trưng quan trọng nhất"); plt.tight_layout()
plt.savefig(f"{FIGS}/04_rf_feature_importance.png", dpi=130); plt.close()

gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
gbr.fit(X_tr, y_tr)
gbr_pred = gbr.predict(X_val)
gbr_rmse = float(np.sqrt(mean_squared_error(y_val, gbr_pred)))
gbr_mae = float(mean_absolute_error(y_val, gbr_pred))
gbr_r2 = float(r2_score(y_val, gbr_pred))
results["gradient_boosting"] = {"rmse": gbr_rmse, "mae": gbr_mae, "r2": gbr_r2}

fig, axes = plt.subplots(1,2, figsize=(13,5.5))
axes[0].scatter(y_val, rf_pred, alpha=0.5, color="darkorange")
axes[0].plot([y_val.min(),y_val.max()],[y_val.min(),y_val.max()],'k--')
axes[0].set_title(f"Random Forest (R²={rf_r2:.3f})")
axes[1].scatter(y_val, gbr_pred, alpha=0.5, color="seagreen")
axes[1].plot([y_val.min(),y_val.max()],[y_val.min(),y_val.max()],'k--')
axes[1].set_title(f"Gradient Boosting (R²={gbr_r2:.3f})")
plt.tight_layout()
plt.savefig(f"{FIGS}/05_pred_vs_actual.png", dpi=130); plt.close()

comp_df = pd.DataFrame({"Model":["Random Forest","Gradient Boosting"],"RMSE":[rf_rmse,gbr_rmse]})
plt.figure(figsize=(7,5))
sns.barplot(data=comp_df, x="Model", y="RMSE", palette=["darkorange","seagreen"])
plt.title("So sánh RMSE giữa 2 mô hình"); plt.tight_layout()
plt.savefig(f"{FIGS}/06_model_comparison.png", dpi=130); plt.close()

best_name = "Gradient Boosting" if gbr_rmse < rf_rmse else "Random Forest"
best = gbr if gbr_rmse < rf_rmse else rf
best.fit(X_train_full, y)
test_pred = best.predict(X_test_final)
pd.DataFrame({"Id": test_ids, "SalePrice": test_pred}).to_csv(f"{OUT}/submission.csv", index=False)
results["best_model"] = best_name

with open(f"{OUT}/results.json","w",encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print("RF:", results["random_forest"])
print("GBR:", results["gradient_boosting"])
print("Best:", best_name)
