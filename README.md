# Solar Power Forecasting

基于光伏电站历史发电与气象数据的短时功率预测项目。项目核心目标是预测下一时刻（t+1）的 `DC_POWER`，并通过时间序列方式完成数据处理、特征工程、模型调参与评估。

## 1. 项目亮点

- 双电站数据融合：同时处理 Plant 1 和 Plant 2 的发电与气象数据
- 时间序列切分：按时间边界切分训练集/测试集/预测集，避免数据泄漏
- 强化特征工程：包含 lag、rolling、差分、交叉项、周期特征等
- 自动化调参：使用 Optuna + XGBoost，并支持断点续跑
- 可复现实验产物：已保存训练报告、最优参数、模型文件与处理后数据

## 2. 数据说明

原始数据位于 `data/raw/`，项目不提供原始数据集和预处理数据集。可以使用项目提供的脚本 `data/raw/raw.py`，通过 KaggleHub 下载数据集（数据来源：anikannal/solar-power-generation-data）。数据集地址：https://www.kaggle.com/datasets/anikannal/solar-power-generation-data。

## 3. 项目结构

```text
solar-power-forecasting/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
├─ src/
│  ├─ make_dataset.py
│  ├─ baseline_model.py
│  └─ predict.py
├─ solar-power-forecasting.ipynb
└─ requirements.txt
```

## 4. 环境准备

### 4.1 Python 版本

建议使用 Python 3.9+。

### 4.2 安装依赖

方式 A（与仓库一致）：

```bash
pip install -r requirements.txt
```

方式 B（推荐最小依赖，安装更快）：

```bash
pip install numpy pandas scikit-learn xgboost optuna
```

## 5. 快速开始

在项目根目录依次执行以下步骤。

### 步骤 1：数据预处理

```bash
python src/make_dataset.py
```

该步骤会：

- 合并发电数据与气象数据
- 清洗异常值并插值缺失气象变量
- 构造时间与周期特征
- 编码 `PLANT_ID` 和 `SOURCE_KEY`
- 生成 `train.csv`、`test.csv`、`predict.csv`、`full_processed.csv`

### 步骤 2：训练与调参

```bash
python src/baseline_model.py
```

说明：

- 默认模式为 `fast`（见 `src/baseline_model.py` 中 `MODE` 变量）
- 可改为 `standard` 获取更充分搜索
- 使用 `TimeSeriesSplit` 做时序交叉验证
- Optuna study 使用 SQLite 存储，支持中断后续跑

### 步骤 3：预测集评估

```bash
python src/predict.py
```

该脚本会加载 `models/xgb_final_optuna_standard.json`，并在 `data/processed/predict.csv` 上评估 t+1 预测效果。评估脚本可更改评估模型，请在`src/predict.py`中修改模型路径。

## 6. 已有实验结果（仓库当前产物）

来自 `models/final_report_optuna_standard.json`：

- 模式：standard
- 特征数：39
- 最优 CV RMSE：282.7874
- Holdout RMSE：673.3416
- Holdout MAE：283.3968
- Holdout R²：0.9532

来自 `data/processed/data_report.json`：

- 全量样本：136,476
- 训练集：107,788
- 测试集：14,344
- 预测集：14,344
- 时间范围：2020-05-15 00:00:00 到 2020-06-17 23:45:00

## 7. 关键输出文件

### 数据产物（`data/processed/`）

- `train.csv`：训练数据
- `test.csv`：测试数据
- `predict.csv`：预测数据
- `full_processed.csv`：完整处理后数据
- `encoders.json`：类别编码映射
- `data_report.json`：数据统计报告

### 模型产物（`models/`）

- `xgb_best_during_optuna_fast.json`
- `xgb_best_during_optuna_standard.json`
- `xgb_final_optuna_standard.json`
- `best_trial_info_fast.json`
- `best_trial_info_standard.json`
- `final_report_optuna_standard.json`

## 8. 方法概览

### 8.1 目标定义

按逆变器分组后，使用 `TARGET_DC_POWER_T1 = DC_POWER.shift(-1)` 作为下一时刻预测目标。

### 8.2 特征工程

- 时序滞后：`LAG_1`、`LAG_2`、`LAG_4`、`LAG_1DAY`
- 滚动统计：rolling mean/std（窗口 4、8）
- 差分特征：功率与辐照差分
- 交叉特征：辐照 × 温度、辐照 × 时间周期项等
- 周期特征：小时/月份的 sin/cos 编码

### 8.3 评估策略

- 时序切分与交叉验证
- 指标：RMSE、MSE、MAE、R²
- 预测值非负截断，并对夜间场景进行输出校正

## 9. 常见问题

### Q1：运行时报缺少原始数据文件？

请先确认 `data/raw/` 下四个 CSV 已存在；若不存在，可使用 `data/raw/raw.py` 下载。

### Q2：为什么 `predict.py` 默认加载 standard 模型？

脚本中模型路径当前固定为 `models/xgb_final_optuna_standard.json`，如你训练的是 fast 模型，请修改该路径。

### Q3：Windows 下路径和报告中的绝对路径不一致怎么办？

报告中的绝对路径来自历史运行环境，不影响你在当前项目中的相对路径执行。

## 10. 后续可优化方向

- 引入多步预测（t+N）与滚动预测评估
- 加入节假日/太阳高度角等先验特征
- 与 LightGBM/CatBoost/时序深度模型做对比
- 增加模型解释性分析（SHAP）与误差分布可视化

---
