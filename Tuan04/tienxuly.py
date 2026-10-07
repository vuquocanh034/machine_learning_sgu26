
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

OUTLIER_IDS = [524, 1299]

# Biến phân loại mà NaN nghĩa là "không có" (theo data_description.txt)
NONE_CAT_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
# Biến số mà NaN nghĩa là 0 (không có tầng hầm / garage / veneer)
ZERO_NUM_COLS = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]
# Biến thứ bậc chất lượng -> mã hóa 0..5
QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
    "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
ORDINAL_MAPS = {
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}


def doc_du_lieu(path_train="train.csv", path_test="test.csv"):
    """Đọc dữ liệu. Giữ nguyên cách đọc như notebook EDA (pd.read_csv mặc định)."""
    return pd.read_csv(path_train), pd.read_csv(path_test)


def loai_mau_bat_thuong(df):
    """Loại 2 bản ghi Id 524 và 1299 đã phát hiện ở mục 5.4 của báo cáo EDA."""
    return df[~df["Id"].isin(OUTLIER_IDS)].reset_index(drop=True)


class AmesCleaner(BaseEstimator, TransformerMixin):
    """Làm sạch + mã hóa thứ bậc + tạo biến mới + log-transform biến lệch.

    fit():  học trung vị LotFrontage theo Neighborhood, mode của các cột thiếu ít,
            và danh sách biến số lệch phải (|skew| > skew_thresh).
    transform(): trả về DataFrame gồm biến số (đã ở dạng số) và biến phân loại danh định.
    """

    def __init__(self, skew_thresh=0.75):
        self.skew_thresh = skew_thresh

    # ---------- bước không cần học tham số ----------
    def _basic(self, X):
        X = X.copy()
        X = X.drop(columns=[c for c in ["Id", "SalePrice"] if c in X.columns])
        X["MSSubClass"] = X["MSSubClass"].astype(str)          # mã nhóm, không phải biến định lượng
        for c in NONE_CAT_COLS:
            X[c] = X[c].fillna("None")
        for c in ZERO_NUM_COLS:
            X[c] = X[c].fillna(0)
        # GarageYrBlt: không có garage -> dùng YearBuilt để không tạo giá trị 0 vô nghĩa
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])
        # mã hóa thứ bậc
        for c in QUAL_COLS:
            X[c] = X[c].map(QUAL_MAP)
        for c, m in ORDINAL_MAPS.items():
            X[c] = X[c].map(m)
        # biến mới (giải quyết một phần trùng thông tin đã nêu ở mục 4.2 báo cáo EDA)
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        X["TotalBath"] = X["FullBath"] + 0.5 * X["HalfBath"] + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
        X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"] + X["WoodDeckSF"]
        X["HouseAge"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
        X["RemodAge"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)
        X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        X["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        X["Has2ndFlr"] = (X["2ndFlrSF"] > 0).astype(int)
        return X

    def fit(self, X, y=None):
        Xb = self._basic(X)
        self.lot_median_ = X.groupby("Neighborhood")["LotFrontage"].median()
        self.lot_global_ = X["LotFrontage"].median()
        cat_cols = Xb.select_dtypes(exclude=[np.number]).columns.tolist()
        self.cat_modes_ = {c: Xb[c].mode().iloc[0] for c in cat_cols}
        num_cols = Xb.select_dtypes(include=[np.number]).columns.tolist()
        self.num_medians_ = {c: Xb[c].median() for c in num_cols}
        sk = Xb[num_cols].skew()
        # chỉ log-transform biến định lượng (nhiều giá trị khác nhau), lệch phải, không âm
        self.skewed_ = [c for c in num_cols
                        if Xb[c].nunique() > 15 and sk[c] > self.skew_thresh
                        and (Xb[c] >= 0).all() and "Yr" not in c and "Year" not in c]
        self.num_cols_ = num_cols
        self.cat_cols_ = cat_cols
        return self

    def transform(self, X):
        X = X.copy()
        # LotFrontage: trung vị theo khu vực (mục 5.2 báo cáo EDA)
        lf = X["Neighborhood"].map(self.lot_median_)
        X["LotFrontage"] = X["LotFrontage"].fillna(lf).fillna(self.lot_global_)
        Xb = self._basic(X)
        for c, m in self.cat_modes_.items():                  # cột thiếu rất ít (vd. Electrical, MSZoning)
            Xb[c] = Xb[c].fillna(m)
        for c, m in self.num_medians_.items():
            Xb[c] = Xb[c].fillna(m)
        for c in self.skewed_:
            Xb[c] = np.log1p(Xb[c])
        return Xb[self.num_cols_ + self.cat_cols_]


def build_preprocessor(scale=True):
    """AmesCleaner -> (StandardScaler cho biến số, OneHot cho biến phân loại).
    Trả về Pipeline; sau bước này mọi đặc trưng là số nên các selector dùng được ngay."""

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
    ct = ColumnTransformer(
        [("num", StandardScaler() if scale else "passthrough", lambda d: d.select_dtypes(include=[np.number]).columns.tolist()),
         ("cat", ohe, lambda d: d.select_dtypes(exclude=[np.number]).columns.tolist())],
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return Pipeline([("clean", AmesCleaner()), ("encode", ct)])
