
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest, VarianceThreshold,
                                       f_regression, mutual_info_regression)
from sklearn.linear_model import LassoCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

SEED = 42


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Bước 1: giữ đặc trưng có |r| với y >= min_corr.
    Bước 2: với mỗi cặp có |r| > max_inter, bỏ biến có |r| với y thấp hơn
    (xử lý trực tiếp hiện tượng trùng thông tin đã nêu ở mục 4.2 báo cáo EDA)."""

    def __init__(self, min_corr=0.10, max_inter=0.90):
        self.min_corr = min_corr
        self.max_inter = max_inter

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        r = X.apply(lambda c: np.corrcoef(c, y)[0, 1] if c.std() > 0 else 0.0).abs().fillna(0)
        cand = r[r >= self.min_corr].sort_values(ascending=False).index.tolist()
        corr = X[cand].corr().abs()
        keep = []
        for c in cand:                       # đi từ biến tương quan với y cao nhất
            if all(corr.loc[c, k] <= self.max_inter for k in keep):
                keep.append(c)
        self.keep_ = keep
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


class _Wrap(BaseEstimator, TransformerMixin):
    """Bọc selector của sklearn để luôn trả về DataFrame (giữ tên cột)."""

    def __init__(self, sel):
        self.sel = sel

    def fit(self, X, y=None):
        self.cols_ = np.asarray(X.columns)
        self.sel.fit(X, y)
        self.support_ = self.sel.get_support()
        self.kept_ = self.cols_[self.support_]
        return self

    def transform(self, X):
        return X.loc[:, self.kept_]


def make_selector(name, k=60):
    """Trả về selector (hoặc None nếu là baseline)."""
    if name == "Baseline (tất cả)":
        return None
    if name == "Variance":
        return _Wrap(VarianceThreshold(threshold=0.01))
    if name == "Correlation":
        return CorrelationFilter(0.10, 0.90)
    if name == "F-test":
        return _Wrap(SelectKBest(f_regression, k=k))
    if name == "Mutual Info":
        return _Wrap(SelectKBest(lambda X, y: mutual_info_regression(X, y, random_state=SEED), k=k))
    if name == "RFE (Ridge)":
        return _Wrap(RFE(Ridge(alpha=10.0), n_features_to_select=k, step=0.1))
    if name == "Lasso":
        return _Wrap(SelectFromModel(LassoCV(cv=5, random_state=SEED, max_iter=20000, alphas=60)))
    if name == "RF importance":
        return _Wrap(SelectFromModel(RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=SEED),
                                     threshold="mean"))
    if name == "XGB importance":
        return _Wrap(SelectFromModel(XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8,
                                                  colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0),
                                     threshold="mean"))
    raise ValueError(name)


SELECTORS = ["Baseline (tất cả)", "Variance", "Correlation", "F-test", "Mutual Info",
             "RFE (Ridge)", "Lasso", "RF importance", "XGB importance"]
GROUP = {"Baseline (tất cả)": "Baseline", "Variance": "Filter", "Correlation": "Filter", "F-test": "Filter",
         "Mutual Info": "Filter", "RFE (Ridge)": "Wrapper", "Lasso": "Embedded",
         "RF importance": "Embedded", "XGB importance": "Embedded"}


class SVRAutoGamma(SVR):
    """SVR (RBF) với gamma = gamma_factor / số đặc trưng.
    Khi Feature Selection làm số đặc trưng p thay đổi, cách này giữ "bán kính" kernel
    tương đương giữa các cấu hình, nên so sánh giữa các selector công bằng hơn gamma cố định."""

    def __init__(self, C=10.0, epsilon=0.05, gamma_factor=0.1):
        super().__init__(kernel="rbf", C=C, epsilon=epsilon, gamma="scale")
        self.gamma_factor = gamma_factor

    def fit(self, X, y, sample_weight=None):
        self.gamma = self.gamma_factor / np.asarray(X).shape[1]
        return super().fit(X, y, sample_weight)


def make_model(name):
    if name == "SVM":
        return SVRAutoGamma(C=10.0, epsilon=0.05, gamma_factor=0.1)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, max_features=0.4, min_samples_leaf=1,
                                     n_jobs=-1, random_state=SEED)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=3, subsample=0.8,
                            colsample_bytree=0.6, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=0)
    raise ValueError(name)


MODELS = ["SVM", "RandomForest", "XGBoost"]


# ======================================================================
# Hàm đánh giá
# ======================================================================
import time
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tien_xu_ly import build_preprocessor


def _metrics(y_true_log, y_pred_log):
    rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2 = float(r2_score(y_true_log, y_pred_log))
    mae = float(mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log)))
    return rmse, r2, mae


def danh_gia_mot_lan(df_tr, y_tr, df_va, y_va, selectors=SELECTORS, models=MODELS, k=60):
    """Fit tiền xử lý + selector trên (df_tr, y_tr), đánh giá trên (df_va, y_va).
    Trả về (list dòng kết quả, dict tên selector -> danh sách đặc trưng được giữ)."""
    prep = build_preprocessor()
    Xtr = prep.fit_transform(df_tr, y_tr)
    Xva = prep.transform(df_va)
    rows, kept = [], {}
    for sname in selectors:
        sel = make_selector(sname, k=k)
        t0 = time.time()
        if sel is None:
            A, B = Xtr, Xva
            kept[sname] = list(Xtr.columns)
        else:
            sel.fit(Xtr, y_tr)
            A, B = sel.transform(Xtr), sel.transform(Xva)
            kept[sname] = list(A.columns)
        t_sel = time.time() - t0
        for mname in models:
            m = make_model(mname)
            t1 = time.time()
            m.fit(A, y_tr)
            t_fit = time.time() - t1
            rmse, r2, mae = _metrics(y_va, m.predict(B))
            rows.append(dict(Selector=sname, Nhom=GROUP[sname], Model=mname, So_dac_trung=A.shape[1],
                             RMSE_log=rmse, R2=r2, MAE_USD=mae, Thoi_gian_chon=t_sel, Thoi_gian_fit=t_fit))
    return rows, kept


def cross_validate_fs(df, y, n_splits=5, seed=SEED, **kw):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_rows, all_kept = [], []
    for i, (a, b) in enumerate(kf.split(df)):
        rows, kept = danh_gia_mot_lan(df.iloc[a], y.iloc[a], df.iloc[b], y.iloc[b], **kw)
        for r in rows:
            r["Fold"] = i + 1
        all_rows += rows
        all_kept.append(kept)
        print(f"  fold {i+1}/{n_splits} xong", flush=True)
    return pd.DataFrame(all_rows), all_kept


def jaccard_on_dinh(all_kept, selector):
    sets = [set(k[selector]) for k in all_kept]
    v = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(v))


def quet_k(df_tr, y_tr, df_va, y_va, ks=(10, 20, 40, 60, 100, 150), methods=("F-test", "Mutual Info", "RF importance"),
           models=MODELS):
    """Xếp hạng đặc trưng một lần rồi lấy top-k, đánh giá từng mô hình."""
    prep = build_preprocessor()
    Xtr = prep.fit_transform(df_tr, y_tr)
    Xva = prep.transform(df_va)
    rankings = {}
    Xa = Xtr.values
    f, _ = f_regression(Xa, y_tr)
    rankings["F-test"] = list(Xtr.columns[np.argsort(-np.nan_to_num(f))])
    mi = mutual_info_regression(Xa, y_tr, random_state=SEED)
    rankings["Mutual Info"] = list(Xtr.columns[np.argsort(-mi)])
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=SEED).fit(Xtr, y_tr)
    rankings["RF importance"] = list(Xtr.columns[np.argsort(-rf.feature_importances_)])
    rows = []
    for meth in methods:
        for k in ks:
            cols = rankings[meth][:k]
            for mname in models:
                m = make_model(mname).fit(Xtr[cols], y_tr)
                rmse, r2, mae = _metrics(y_va, m.predict(Xva[cols]))
                rows.append(dict(Selector=meth, k=k, Model=mname, RMSE_log=rmse, R2=r2, MAE_USD=mae))
    return pd.DataFrame(rows), rankings
