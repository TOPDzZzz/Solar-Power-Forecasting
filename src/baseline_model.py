# name=src/baseline_model.py
import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 配置
# =========================
MODE = "fast"   # "fast" or "standard"

if MODE == "fast":
    N_TRIALS = 15
    N_SPLITS = 3
    OPTUNA_N_ESTIMATORS = 700
    OPTUNA_EARLY_STOP = 30
    FINAL_N_ESTIMATORS = 1200
    FINAL_EARLY_STOP = 50
else:
    N_TRIALS = 40
    N_SPLITS = 5
    OPTUNA_N_ESTIMATORS = 1800
    OPTUNA_EARLY_STOP = 80
    FINAL_N_ESTIMATORS = 2500
    FINAL_EARLY_STOP = 100


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["SOURCE_KEY_ENCODED", "DATE_TIME"]).reset_index(drop=True)
    g = df.groupby("SOURCE_KEY_ENCODED")

    df["TARGET_DC_POWER_T1"] = g["DC_POWER"].shift(-1)

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


def load_data(processed_dir):
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("找不到 train.csv 或 test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for d in [train_df, test_df]:
        d["DATE_TIME"] = pd.to_datetime(d["DATE_TIME"], errors="coerce")

    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
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
        "PLANT_ID_ENCODED", "SOURCE_KEY_ENCODED", "IRRADIATION",
        "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE",
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
    df = df.dropna(subset=need_cols).copy()

    unique_ts = np.array(sorted(df["DATE_TIME"].unique()))
    split_idx = int(len(unique_ts) * 0.9)
    split_t = unique_ts[split_idx]

    train_data = df[df["DATE_TIME"] < split_t].copy()
    test_data = df[df["DATE_TIME"] >= split_t].copy()

    return train_data, test_data, selected_features, split_t


def evaluate_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    processed_dir = os.path.join(project_root, "data", "processed")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 输出文件
    best_model_path = os.path.join(model_dir, f"xgb_best_during_optuna_{MODE}.json")
    best_info_path = os.path.join(model_dir, f"best_trial_info_{MODE}.json")
    final_model_path = os.path.join(model_dir, f"xgb_final_optuna_{MODE}.json")
    final_report_path = os.path.join(model_dir, f"final_report_optuna_{MODE}.json")

    # Optuna 持久化数据库（支持中断续跑）
    study_db_path = os.path.join(model_dir, f"optuna_study_{MODE}.db")
    storage_url = f"sqlite:///{study_db_path}"
    study_name = f"xgb_optuna_resume_{MODE}"

    train_data, test_data, selected_features, split_t = load_data(processed_dir)

    X_train = train_data[selected_features].reset_index(drop=True)
    y_train = train_data["TARGET_DC_POWER_T1"].astype(float).reset_index(drop=True)

    X_test = test_data[selected_features].reset_index(drop=True)
    y_test = test_data["TARGET_DC_POWER_T1"].astype(float).reset_index(drop=True)

    print(f"MODE={MODE}")
    print(f"Train={X_train.shape}, Test={X_test.shape}, n_features={X_train.shape[1]}")
    print(f"Study DB: {study_db_path}")

    # 读取已有 best（支持重启后继续保存/替换 best 模型）
    if os.path.exists(best_info_path):
        with open(best_info_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        best_rmse_so_far = float(old.get("best_cv_rmse", float("inf")))
        best_trial_so_far = int(old.get("best_trial", -1))
    else:
        best_rmse_so_far = float("inf")
        best_trial_so_far = -1

    state = {
        "best_rmse": best_rmse_so_far,
        "best_trial": best_trial_so_far,
        "best_params": None
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial: optuna.Trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "n_estimators": OPTUNA_N_ESTIMATORS,

            # 重点搜索
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),

            # 辅助参数
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        }

        fold_rmses = []
        last_fold_model = None

        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model = xgb.XGBRegressor(**params, early_stopping_rounds=OPTUNA_EARLY_STOP)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            pred = model.predict(X_va)
            pred = np.clip(pred, 0, None)
            rmse = np.sqrt(mean_squared_error(y_va, pred))
            fold_rmses.append(rmse)
            last_fold_model = model

        cv_rmse = float(np.mean(fold_rmses))

        # 每轮写入中间值，便于追踪
        trial.set_user_attr("cv_rmse", cv_rmse)

        # 更优则立即覆盖best模型与best信息
        if cv_rmse < state["best_rmse"] and last_fold_model is not None:
            state["best_rmse"] = cv_rmse
            state["best_trial"] = trial.number
            state["best_params"] = params

            last_fold_model.save_model(best_model_path)

            best_info = {
                "mode": MODE,
                "study_name": study_name,
                "best_trial": trial.number,
                "best_cv_rmse": cv_rmse,
                "params": params
            }
            with open(best_info_path, "w", encoding="utf-8") as f:
                json.dump(best_info, f, ensure_ascii=False, indent=2)

            print(f"[Trial {trial.number}] ✅ New Best CV RMSE={cv_rmse:.4f} -> saved best model")
        else:
            print(f"[Trial {trial.number}] CV RMSE={cv_rmse:.4f} (best={state['best_rmse']:.4f})")

        return cv_rmse

    # 关键：load_if_exists=True 支持断点续跑
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize"
    )

    done_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"已完成 trial 数: {done_trials}")
    print(f"本次计划新增 trial 数: {N_TRIALS}")

    study.optimize(
        objective,
        n_trials=N_TRIALS,           # 每次运行新增N_TRIALS个
        show_progress_bar=True,
        gc_after_trial=True
    )

    # 取全历史最佳（包括之前中断前的）
    best_trial = study.best_trial
    best_value = float(study.best_value)
    best_params = study.best_params

    print("\nOptuna完成（含续跑）")
    print(f"Total trials in study: {len(study.trials)}")
    print(f"Best trial: {best_trial.number}")
    print(f"Best CV RMSE: {best_value:.4f}")
    print(f"Best params: {best_params}")

    # 用全历史最佳参数做最终模型
    final_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": FINAL_N_ESTIMATORS,
        **best_params
    }

    n = len(X_train)
    val_idx = int(n * 0.9)
    X_tr, X_val = X_train.iloc[:val_idx], X_train.iloc[val_idx:]
    y_tr, y_val = y_train.iloc[:val_idx], y_train.iloc[val_idx:]

    final_model = xgb.XGBRegressor(**final_params, early_stopping_rounds=FINAL_EARLY_STOP)
    final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = final_model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    # 与你原逻辑保持一致
    if "solar_proxy" in X_test.columns:
        night_mask = X_test["solar_proxy"] <= 0.01
        y_pred[night_mask] = 0.0

    metrics = evaluate_metrics(y_test, y_pred)

    print("\n======== Final Holdout ========")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MSE : {metrics['mse']:.4f}")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"R2  : {metrics['r2']:.4f}")
    print("================================")

    final_model.save_model(final_model_path)

    report = {
        "mode": MODE,
        "study": {
            "study_name": study_name,
            "storage": storage_url,
            "total_trials": len(study.trials),
            "added_trials_this_run": N_TRIALS,
            "best_trial": int(best_trial.number),
            "best_cv_rmse": best_value,
            "best_params": best_params
        },
        "split_time": str(split_t),
        "n_features": int(X_train.shape[1]),
        "holdout_metrics": metrics,
        "artifacts": {
            "best_model_during_search": best_model_path,
            "best_trial_info": best_info_path,
            "final_model": final_model_path,
            "study_db": study_db_path
        }
    }

    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n已保存：{best_model_path}")
    print(f"已保存：{best_info_path}")
    print(f"已保存：{final_model_path}")
    print(f"已保存：{final_report_path}")
    print(f"Study数据库：{study_db_path}")


if __name__ == "__main__":
    main()