import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_paths():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    return project_root, raw_dir, processed_dir


def read_raw_data(raw_dir):
    print("正在从 data/raw 加载数据...")
    p1_gen = pd.read_csv(os.path.join(raw_dir, "Plant_1_Generation_Data.csv"))
    p1_wea = pd.read_csv(os.path.join(raw_dir, "Plant_1_Weather_Sensor_Data.csv"))
    p2_gen = pd.read_csv(os.path.join(raw_dir, "Plant_2_Generation_Data.csv"))
    p2_wea = pd.read_csv(os.path.join(raw_dir, "Plant_2_Weather_Sensor_Data.csv"))
    return p1_gen, p1_wea, p2_gen, p2_wea


def parse_datetime_by_name(df, name):
    """
    按文件类型指定时间格式，避免 dayfirst 警告和潜在误解析
    """
    if name == "p1_gen":
        # Plant_1_Generation_Data 常见格式: 15-05-2020 00:00
        df["DATE_TIME"] = pd.to_datetime(
            df["DATE_TIME"], format="%d-%m-%Y %H:%M", errors="coerce"
        )
    else:
        # 其余通常是: 2020-05-15 00:00:00
        df["DATE_TIME"] = pd.to_datetime(
            df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

    nat_ratio = df["DATE_TIME"].isna().mean()
    print(f"{name} DATE_TIME NaT ratio: {nat_ratio:.4%}")
    return df


def preprocess_merge(p1_gen, p1_wea, p2_gen, p2_wea):
    # 1) 解析时间
    p1_gen = parse_datetime_by_name(p1_gen, "p1_gen")
    p1_wea = parse_datetime_by_name(p1_wea, "p1_wea")
    p2_gen = parse_datetime_by_name(p2_gen, "p2_gen")
    p2_wea = parse_datetime_by_name(p2_wea, "p2_wea")

    # 2) 去掉无效时间
    p1_gen = p1_gen.dropna(subset=["DATE_TIME"]).copy()
    p1_wea = p1_wea.dropna(subset=["DATE_TIME"]).copy()
    p2_gen = p2_gen.dropna(subset=["DATE_TIME"]).copy()
    p2_wea = p2_wea.dropna(subset=["DATE_TIME"]).copy()

    # 3) 去掉天气中冲突列
    p1_wea = p1_wea.drop(columns=["PLANT_ID", "SOURCE_KEY"], errors="ignore")
    p2_wea = p2_wea.drop(columns=["PLANT_ID", "SOURCE_KEY"], errors="ignore")

    # 4) 以发电数据为主 left merge
    p1_df = pd.merge(p1_gen, p1_wea, on="DATE_TIME", how="left")
    p2_df = pd.merge(p2_gen, p2_wea, on="DATE_TIME", how="left")

    # 5) 合并
    df = pd.concat([p1_df, p2_df], axis=0, ignore_index=True)

    # 6) 去重并重置索引（关键！）
    df = df.drop_duplicates(subset=["DATE_TIME", "PLANT_ID", "SOURCE_KEY"]).reset_index(drop=True)

    return df


def clean_and_feature_engineering(df):
    print("正在进行数据清洗与特征工程...")

    # 数值化
    numeric_cols = [
        "DC_POWER", "AC_POWER",
        "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 物理异常: 负值设为 NaN
    for c in ["IRRADIATION", "DC_POWER", "AC_POWER"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # 排序 + 重置索引（避免重复标签问题）
    df = df.sort_values(["PLANT_ID", "DATE_TIME"]).reset_index(drop=True)

    # 用 transform 做插值，返回长度与原df一致，避免 apply 对齐问题
    weather_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df.groupby("PLANT_ID")[col].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both")
            )

    # 去掉目标缺失
    df = df.dropna(subset=["DC_POWER", "AC_POWER"]).copy()

    # 时间特征
    df["hour"] = df["DATE_TIME"].dt.hour
    df["minute"] = df["DATE_TIME"].dt.minute
    df["day"] = df["DATE_TIME"].dt.day
    df["month"] = df["DATE_TIME"].dt.month
    df["dayofweek"] = df["DATE_TIME"].dt.dayofweek

    # 周期特征
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["is_daytime"] = (df["IRRADIATION"] > 0).astype(int)

    # 类别编码
    le_source = LabelEncoder()
    le_plant = LabelEncoder()

    df["SOURCE_KEY"] = df["SOURCE_KEY"].astype(str)
    df["PLANT_ID"] = df["PLANT_ID"].astype(str)

    df["SOURCE_KEY_ENCODED"] = le_source.fit_transform(df["SOURCE_KEY"])
    df["PLANT_ID_ENCODED"] = le_plant.fit_transform(df["PLANT_ID"])

    # 保留列
    keep_cols = [
        "DATE_TIME",
        "PLANT_ID", "PLANT_ID_ENCODED",
        "SOURCE_KEY", "SOURCE_KEY_ENCODED",
        "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
        "hour", "minute", "day", "month", "dayofweek",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "is_daytime",
        "DC_POWER", "AC_POWER",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # 最终排序
    df = df.sort_values(
        by=["DATE_TIME", "PLANT_ID_ENCODED", "SOURCE_KEY_ENCODED"]
    ).reset_index(drop=True)

    return df, le_source, le_plant


def split_by_time_boundary(df, train_ratio=0.8, test_ratio=0.1):
    unique_ts = np.array(sorted(df["DATE_TIME"].unique()))
    n_ts = len(unique_ts)
    if n_ts < 10:
        raise ValueError("唯一时间戳过少，无法稳定切分。")

    train_idx = int(n_ts * train_ratio)
    test_idx = int(n_ts * (train_ratio + test_ratio))

    train_t = unique_ts[train_idx]
    test_t = unique_ts[test_idx]

    train_df = df[df["DATE_TIME"] < train_t].copy()
    test_df = df[(df["DATE_TIME"] >= train_t) & (df["DATE_TIME"] < test_t)].copy()
    predict_df = df[df["DATE_TIME"] >= test_t].copy()

    return train_df, test_df, predict_df


def save_outputs(processed_dir, train_df, test_df, predict_df, full_df, le_source, le_plant):
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    predict_df.to_csv(os.path.join(processed_dir, "predict.csv"), index=False)
    full_df.to_csv(os.path.join(processed_dir, "full_processed.csv"), index=False)

    encoders = {
        "PLANT_ID_classes": le_plant.classes_.tolist(),
        "SOURCE_KEY_classes": le_source.classes_.tolist(),
    }
    with open(os.path.join(processed_dir, "encoders.json"), "w", encoding="utf-8") as f:
        json.dump(encoders, f, ensure_ascii=False, indent=2)

    report = {
        "rows_full": int(len(full_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "rows_predict": int(len(predict_df)),
        "time_min": str(full_df["DATE_TIME"].min()),
        "time_max": str(full_df["DATE_TIME"].max()),
        "missing_rate": full_df.isna().mean().round(6).to_dict(),
    }
    with open(os.path.join(processed_dir, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n数据划分完成:")
    print(f" - 训练集: {len(train_df)}")
    print(f" - 测试集: {len(test_df)}")
    print(f" - 预测集: {len(predict_df)}")
    print(f"\n已保存到: {processed_dir}")


def main():
    _, raw_dir, processed_dir = build_paths()
    p1_gen, p1_wea, p2_gen, p2_wea = read_raw_data(raw_dir)
    df = preprocess_merge(p1_gen, p1_wea, p2_gen, p2_wea)
    df, le_source, le_plant = clean_and_feature_engineering(df)
    train_df, test_df, predict_df = split_by_time_boundary(df, train_ratio=0.8, test_ratio=0.1)
    save_outputs(processed_dir, train_df, test_df, predict_df, df, le_source, le_plant)


if __name__ == "__main__":
    main()