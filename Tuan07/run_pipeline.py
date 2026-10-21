import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              confusion_matrix, classification_report)

DATA = "/home/claude/mg/data"
FIGS = "/home/claude/mg/figs"
OUT = "/home/claude/mg/outputs"
os.makedirs(FIGS, exist_ok=True); os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
results = {"train_shape": train.shape, "test_shape": test.shape}

GENRE_NAMES = {0:"Acoustic/Folk",1:"Alt_Music",2:"Blues",3:"Bollywood",4:"Country",
               5:"HipHop",6:"Indie",7:"Instrumental",8:"Metal",9:"Pop",10:"Rock"}
train["GenreName"] = train["Class"].map(GENRE_NAMES)

# ---- EDA ----
plt.figure(figsize=(10,5))
order = train["Class"].value_counts().sort_index().index
sns.countplot(data=train, x="Class", order=order, color="teal")
plt.title("Phân phối số lượng bài hát theo thể loại (Class)")
plt.xlabel("Class"); plt.ylabel("Số lượng")
plt.tight_layout(); plt.savefig(f"{FIGS}/01_class_dist.png", dpi=130); plt.close()
results["class_dist"] = train["Class"].value_counts().sort_index().to_dict()

plt.figure(figsize=(10,6))
sns.heatmap(train.isnull(), cbar=False, cmap="viridis")
plt.title("Bản đồ giá trị thiếu - Train")
plt.tight_layout(); plt.savefig(f"{FIGS}/02_missing_train.png", dpi=130); plt.close()
results["missing_train"] = train.isnull().sum()[train.isnull().sum()>0].to_dict()

num_cols = ["Popularity","danceability","energy","key","loudness","mode","speechiness",
            "acousticness","instrumentalness","liveness","valence","tempo",
            "duration_in min/ms","time_signature"]
corr = train[num_cols+["Class"]].corr()
plt.figure(figsize=(11,9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Ma trận tương quan giữa các đặc trưng âm thanh")
plt.tight_layout(); plt.savefig(f"{FIGS}/03_corr_heatmap.png", dpi=130); plt.close()

# ---- Data Preparation ----
# Chuẩn hoá duration (một số dòng là phút, một số là ms -> quy đổi theo ngưỡng)
def fix_duration(df):
    df = df.copy()
    mask = df["duration_in min/ms"] < 1000  # giá trị nhỏ => đang ở dạng phút
    df.loc[mask, "duration_in min/ms"] = df.loc[mask, "duration_in min/ms"] * 60000
    df = df.rename(columns={"duration_in min/ms": "duration_ms"})
    return df

train_feat = fix_duration(train.drop(columns=["GenreName"]))
test_feat = fix_duration(test)

drop_cols = ["Id", "Artist Name", "Track Name"]
y = train_feat["Class"].copy()
test_ids = test_feat["Id"].copy()
X_train_full = train_feat.drop(columns=drop_cols + ["Class"], errors="ignore")
X_test_final = test_feat.drop(columns=drop_cols, errors="ignore")

# Điền missing bằng median (numeric) dựa trên tập train
for c in X_train_full.columns:
    if X_train_full[c].isnull().any():
        med = X_train_full[c].median()
        X_train_full[c] = X_train_full[c].fillna(med)
        X_test_final[c] = X_test_final[c].fillna(med)

results["n_features"] = X_train_full.shape[1]
results["X_train_shape"] = X_train_full.shape
results["X_test_shape"] = X_test_final.shape

X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y, test_size=0.2, random_state=42, stratify=y)

# ---- Modeling ----
rf = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_tr, y_tr)
rf_pred = rf.predict(X_val)
rf_acc = float(accuracy_score(y_val, rf_pred))
rf_f1 = float(f1_score(y_val, rf_pred, average="macro"))
rf_prec = float(precision_score(y_val, rf_pred, average="macro", zero_division=0))
rf_rec = float(recall_score(y_val, rf_pred, average="macro", zero_division=0))
results["random_forest"] = {"accuracy": rf_acc, "f1_macro": rf_f1, "precision_macro": rf_prec, "recall_macro": rf_rec}

importances = pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
plt.figure(figsize=(9,6))
sns.barplot(x=importances.values, y=importances.index, color="steelblue")
plt.title("Random Forest - Mức độ quan trọng của đặc trưng")
plt.tight_layout(); plt.savefig(f"{FIGS}/04_rf_feature_importance.png", dpi=130); plt.close()

gbc = GradientBoostingClassifier(n_estimators=300, learning_rate=0.08, max_depth=3, random_state=42)
gbc.fit(X_tr, y_tr)
gbc_pred = gbc.predict(X_val)
gbc_acc = float(accuracy_score(y_val, gbc_pred))
gbc_f1 = float(f1_score(y_val, gbc_pred, average="macro"))
gbc_prec = float(precision_score(y_val, gbc_pred, average="macro", zero_division=0))
gbc_rec = float(recall_score(y_val, gbc_pred, average="macro", zero_division=0))
results["gradient_boosting"] = {"accuracy": gbc_acc, "f1_macro": gbc_f1, "precision_macro": gbc_prec, "recall_macro": gbc_rec}

# Confusion matrix (best model)
best_name = "Gradient Boosting" if gbc_acc > rf_acc else "Random Forest"
best_pred = gbc_pred if gbc_acc > rf_acc else rf_pred
best_model_val = gbc if gbc_acc > rf_acc else rf
cm = confusion_matrix(y_val, best_pred)
plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Ma trận nhầm lẫn - {best_name} (validation)")
plt.xlabel("Dự đoán"); plt.ylabel("Thực tế")
plt.tight_layout(); plt.savefig(f"{FIGS}/05_confusion_matrix.png", dpi=130); plt.close()

comp_df = pd.DataFrame({"Model":["Random Forest","Gradient Boosting"], "Accuracy":[rf_acc,gbc_acc], "F1-macro":[rf_f1,gbc_f1]})
comp_melt = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
plt.figure(figsize=(7,5))
sns.barplot(data=comp_melt, x="Model", y="Value", hue="Metric", palette=["darkorange","seagreen"])
plt.title("So sánh Accuracy & F1-macro giữa 2 mô hình"); plt.ylim(0,1)
plt.tight_layout(); plt.savefig(f"{FIGS}/06_model_comparison.png", dpi=130); plt.close()

results["best_model"] = best_name

# ---- Deployment: retrain on full train, predict test ----
best_model = gbc if gbc_acc > rf_acc else rf
best_model.fit(X_train_full, y)
test_pred = best_model.predict(X_test_final)
pd.DataFrame({"Id": test_ids, "Class": test_pred}).to_csv(f"{OUT}/submission.csv", index=False)

with open(f"{OUT}/results.json","w",encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print("RF:", results["random_forest"])
print("GBC:", results["gradient_boosting"])
print("Best:", best_name)
