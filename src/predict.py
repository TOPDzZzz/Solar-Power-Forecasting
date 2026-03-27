import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["SOURCE_KEY_ENCODED", "DATE_TIME"]).reset_index(drop=True)
    g = df.groupby("SOURCE_KEY_ENCODED")

    # t+1 评估目标
    df["TARGET_DC_POWER_T1"] = g["DC_POWER"].shift(-1)

    # 与训练保持一致的特征
    df["DC_POWER_LAG_1"] = g["DC_POWER"].shift(1)
    df["DC_POWER_LAG_2"] = g["DC_POWER"].shift(2)
    df["DC_POWER_LAG_4"] = g["DC_POWER"].shift(4)

    df["IRRADIATION_LAG_1"] = g["IRRADIATION"].shift(1)
    df["IRRADIATION_LAG_2"] = g["IRRADIATION"].shift(2)
    df["IRRADIATION_LAG_4"] = g["IRRADIATION"].shift(4)

    df["MODULE_TEMP_LAG_1"] = g["MODULE_TEMPERATURE"].shift(1)
    df["AMBIENT_TEMP_LAG_1"] = g["AMBIENT_TEMPERATURE"].shift(1)

    df["DC_POWER_ROLL_MEAN_4"] = g["DC_POWER"].transform(lambda s: s.shift(1).rolling(4).mean())
    df["DC_POWER_ROLL_STD_4"] = g["DC_POWER"].transform(lambda s: s.shift(1).rolling(4).std())
    df["IRRADIATION_ROLL_MEAN_4"] = g["IRRADIATION"].transform(lambda s: s.shift(1).rolling(4).mean())
    df["IRRADIATION_ROLL_STD_4"] = g["IRRADIATION"].transform(lambda s: s.shift(1).rolling(4).std())

    df["DC_POWER_DIFF_1"] = g["DC_POWER"].diff(1)
    df["IRRADIATION_DIFF_1"] = g["IRRADIATION"].diff(1)

    df["TEMP_DIFF"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
    df["IRR_X_TEMP"] = df["IRRADIATION"] * df["MODULE_TEMPERATURE"]

    df["solar_proxy"] = np.maximum(0, np.sin((df["hour"] - 6) / 12 * np.pi))
    df["TEMP_RATIO"] = df["MODULE_TEMPERATURE"] / (df["AMBIENT_TEMPERATURE"] + 1e-3)
    df["TEMP_SQ"] = df["MODULE_TEMPERATURE"] ** 2
    df["IRRADIATION_SQ"] = df["IRRADIATION"] ** 2
    df["IRRADIATION_SQRT"] = np.sqrt(np.clip(df["IRRADIATION"], 0, None))

    df["IRR_X_SOLAR_PROXY"] = df["IRRADIATION"] * df["solar_proxy"]
    df["IRR_X_TEMP_DIFF"] = df["IRRADIATION"] * (df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"])
    df["IRR_X_HOUR_COS"] = df["IRRADIATION"] * df["hour_cos"]

    df["DC_POWER_ROLL_MEAN_8"] = g["DC_POWER"].transform(lambda s: s.shift(1).rolling(8).mean())
    df["IRRADIATION_ROLL_MEAN_8"] = g["IRRADIATION"].transform(lambda s: s.shift(1).rolling(8).mean())
    df["IRRADIATION_ROLL_STD_8"] = g["IRRADIATION"].transform(lambda s: s.shift(1).rolling(8).std())

    df["month"] = df["DATE_TIME"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["DC_POWER_LAG_1DAY"] = g["DC_POWER"].shift(96)

    for c in ["DC_POWER_ROLL_STD_4", "IRRADIATION_ROLL_STD_4", "IRRADIATION_ROLL_STD_8"]:
        df[c] = df[c].fillna(0.0)

    return df


def main():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    processed_dir = os.path.join(project_root, "data", "processed")
    model_dir = os.path.join(project_root, "models")

    predict_path = os.path.join(processed_dir, "predict.csv")
    model_path = os.path.join(model_dir, "xgb_final_optuna_standard.json")  # 按你的实际模型名改

    if not os.path.exists(predict_path):
        raise FileNotFoundError(f"找不到文件: {predict_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型: {model_path}")

    df = pd.read_csv(predict_path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    df = df.dropna(subset=["DATE_TIME"]).copy()

    numeric_cols = [
        "DC_POWER", "AC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE",
        "PLANT_ID_ENCODED", "SOURCE_KEY_ENCODED", "hour", "hour_sin", "hour_cos"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = add_features(df)

    selected_features = [
        "PLANT_ID_ENCODED",
        "SOURCE_KEY_ENCODED",
        "IRRADIATION",
        "MODULE_TEMPERATURE",
        "AMBIENT_TEMPERATURE",
        "hour", "hour_sin", "hour_cos",
        "DC_POWER_LAG_1", "DC_POWER_LAG_2", "DC_POWER_LAG_4",
        "IRRADIATION_LAG_1", "IRRADIATION_LAG_2", "IRRADIATION_LAG_4",
        "MODULE_TEMP_LAG_1", "AMBIENT_TEMP_LAG_1",
        "DC_POWER_ROLL_MEAN_4", "DC_POWER_ROLL_STD_4",
        "IRRADIATION_ROLL_MEAN_4", "IRRADIATION_ROLL_STD_4",
        "DC_POWER_DIFF_1", "IRRADIATION_DIFF_1",
        "TEMP_DIFF", "IRR_X_TEMP",
        "solar_proxy", "TEMP_RATIO", "TEMP_SQ",
        "IRRADIATION_SQ", "IRRADIATION_SQRT",
        "IRR_X_SOLAR_PROXY", "IRR_X_TEMP_DIFF", "IRR_X_HOUR_COS",
        "DC_POWER_ROLL_MEAN_8", "IRRADIATION_ROLL_MEAN_8", "IRRADIATION_ROLL_STD_8",
        "month", "month_sin", "month_cos", "DC_POWER_LAG_1DAY"
    ]
    selected_features = [c for c in selected_features if c in df.columns]

    need_cols = ["TARGET_DC_POWER_T1"] + selected_features
    eval_df = df.dropna(subset=need_cols).copy()

    if len(eval_df) == 0:
        raise ValueError("可评估样本为0，请检查 predict.csv 是否包含足够历史行用于lag/rolling特征。")

    X_eval = eval_df[selected_features]
    y_true = eval_df["TARGET_DC_POWER_T1"].astype(float).values

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    y_pred = model.predict(X_eval)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    # 夜间纠偏（与训练评估一致）
    if "solar_proxy" in X_eval.columns:
        night_mask = X_eval["solar_proxy"] <= 0.01
        y_pred[night_mask] = 0.0

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    print("======== Predict.csv 评估结果（t+1） ========")
    print(f"样本数: {len(eval_df)}")
    print(f"特征数: {X_eval.shape[1]}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")
    print("============================================")


if __name__ == "__main__":
    main()