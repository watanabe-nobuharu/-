from google.colab import drive
drive.mount('/content/drive')

# 基本的なライブラリは、Colabではデフォルトで利用可能

import pandas as pd               # データを表のように扱うライブラリ
import numpy as np                # 数値計算を速くするライブラリ
import seaborn as sns             # きれいなグラフを簡単に作るライブラリ
import matplotlib.pyplot as plt   # グラフを作る基本的なライブラリ
%matplotlib inline
from sklearn.model_selection import train_test_split  # データを訓練用と検証用に分ける
from sklearn.metrics import mean_squared_log_error # 評価の計算を行うライブラリ
import lightgbm as lgb # 予測モデルに関するライブラリ
!pip install optuna
import optuna
np.random.seed(46)
from re import sub
from lightgbm import LGBMRegressor
import warnings
warnings.simplefilter('ignore')  # 不要な警告を表示しない

# 予測モデルを訓練するためのデータセット
train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/SMBC(Carbon)/data/train.csv', index_col=0)
# 予測モデルに推論（予測)させるデータセット
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/SMBC(Carbon)/data/test.csv', index_col=0)

def describe_data(data, name):
    print(f"\n{name} のデータ概要:\n")
    display(data.info())
    display(data.describe())

describe_data(train, 'train')

describe_data(test, 'test')

GHG_train_10 = train['GHG_Direct_Emissions_10_in_metric_tons'].median()
GHG_train_11 = train['GHG_Direct_Emissions_11_in_metric_tons'].median()
GHG_train_12 = train['GHG_Direct_Emissions_12_in_metric_tons'].median()
GHG_train_13 = train['GHG_Direct_Emissions_13_in_metric_tons'].median()
GHG_test_10 = test['GHG_Direct_Emissions_10_in_metric_tons'].median()
GHG_test_11 = test['GHG_Direct_Emissions_11_in_metric_tons'].median()
GHG_test_12 = test['GHG_Direct_Emissions_12_in_metric_tons'].median()
GHG_test_13 = test['GHG_Direct_Emissions_13_in_metric_tons'].median()

GHG_diff_10 = GHG_test_10 / GHG_train_10
GHG_diff_11 = GHG_test_11 / GHG_train_11
GHG_diff_12 = GHG_test_12 / GHG_train_12
GHG_diff_13 = GHG_test_13 / GHG_train_13
GHG_diff_all = (GHG_diff_10+GHG_diff_11+GHG_diff_12+GHG_diff_13)/4

train['GHG_Direct_Emissions_10_in_metric_tons'] = train['GHG_Direct_Emissions_10_in_metric_tons'] * GHG_diff_10
train['GHG_Direct_Emissions_11_in_metric_tons'] = train['GHG_Direct_Emissions_11_in_metric_tons'] * GHG_diff_11
train['GHG_Direct_Emissions_12_in_metric_tons'] = train['GHG_Direct_Emissions_12_in_metric_tons'] * GHG_diff_12
train['GHG_Direct_Emissions_13_in_metric_tons'] = train['GHG_Direct_Emissions_13_in_metric_tons'] * GHG_diff_13
train['GHG_Direct_Emissions_14_in_metric_tons'] = train['GHG_Direct_Emissions_14_in_metric_tons'] * GHG_diff_all

numerical_features02 = ['GHG_Direct_Emissions_10_in_metric_tons', 'GHG_Direct_Emissions_11_in_metric_tons',
                        'GHG_Direct_Emissions_12_in_metric_tons', 'GHG_Direct_Emissions_13_in_metric_tons']



train['GHG_diff_10_13'] = train['GHG_Direct_Emissions_13_in_metric_tons'] - train['GHG_Direct_Emissions_10_in_metric_tons']
train['GHG_diff_11_13'] = train['GHG_Direct_Emissions_13_in_metric_tons'] - train['GHG_Direct_Emissions_11_in_metric_tons']
train['GHG_diff_12_13'] = train['GHG_Direct_Emissions_13_in_metric_tons'] - train['GHG_Direct_Emissions_12_in_metric_tons']
train['GHG_diff_13_14'] = train['GHG_Direct_Emissions_14_in_metric_tons'] / train['GHG_Direct_Emissions_13_in_metric_tons']
test['GHG_diff_10_13'] = test['GHG_Direct_Emissions_13_in_metric_tons'] - test['GHG_Direct_Emissions_10_in_metric_tons']
test['GHG_diff_11_13'] = test['GHG_Direct_Emissions_13_in_metric_tons'] - test['GHG_Direct_Emissions_11_in_metric_tons']
test['GHG_diff_12_13'] = test['GHG_Direct_Emissions_13_in_metric_tons'] - test['GHG_Direct_Emissions_12_in_metric_tons']

numerical_features01 = [
    'TRI_Air_Emissions_10_in_lbs','TRI_Air_Emissions_11_in_lbs','TRI_Air_Emissions_12_in_lbs','TRI_Air_Emissions_13_in_lbs',
    ]

numerical_features03 = ['GHG_diff_10_13','GHG_diff_11_13','GHG_diff_12_13']

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
for col_list in numerical_features02,numerical_features01:
    for col in col_list:  # Iterate over individual columns in the list
        # Reshape the data to be 2-dimensional using [[ ]] for both train and test data
        scaler.fit(pd.concat([train[[col]], test[[col]]]))
        train[col] = scaler.transform(train[[col]])  # Transform using [[ ]]
        test[col] = scaler.transform(test[[col]])

numerical_features1= ['State','IndustryType']
numerical_features2=['Latitude','Longitude','FIPScode','PrimaryNAICS',
                     ]

# Label-Count Encodingの適用
for col in numerical_features1:
    train[col] = train[col].map(train[col].value_counts().rank(ascending=False, method='first'))
    test[col] = test[col].map(test[col].value_counts().rank(ascending=False, method='first'))

# 訓練用データセットからターゲットを分離する
X = train[numerical_features01+numerical_features02+numerical_features03+numerical_features1+numerical_features2]
y = train['GHG_Direct_Emissions_14_in_metric_tons']

# 投稿のためのテストデータも同様の処理を行う
test_1st = test[numerical_features01+numerical_features02+numerical_features03+numerical_features1+numerical_features2]

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, root_mean_squared_log_error

random_state = 42
# クロスバリデーションの設定
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

# RMSLEスコアリング関数を作成
rmsle_scorer = make_scorer(root_mean_squared_log_error, greater_is_better=False)

# 訓練用データセットを訓練用と検証用に分割する
X_train_1st, X_valid_1st, y_train_1st, y_valid_1st = train_test_split(X, y, test_size=0.1,shuffle = True, random_state=random_state)
# 結果の確認（データフレームの形状）
print(f"X_train: {X_train_1st.shape}, X_valid: {X_valid_1st.shape}")
print(f"y_train: {y_train_1st.shape}, y_valid: {y_valid_1st.shape}")
print(f"test_1st: {test_1st.shape}")

def run1(trial):
    num_leaves = trial.suggest_int('num_leaves', 49, 91)
    max_depth = trial.suggest_int('max_depth', 83, 179)
    learning_rate = trial.suggest_float('learning_rate', 0.052, 0.062)
    n_estimators = trial.suggest_int('n_estimators', 496, 663)
    min_child_samples = trial.suggest_int('min_child_samples', 37, 59)
    min_split_gain = trial.suggest_float('min_split_gain', 0.64, 0.67)
    min_child_weight = trial.suggest_int('min_child_weight', 49, 63)
    subsample = trial.suggest_float('subsample', 0.595, 0.799)
    subsample_freq = trial.suggest_int('subsample_freq', 4, 6)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.704, 0.796)
    reg_alpha = trial.suggest_float('reg_alpha', 0.465, 0.701)
    reg_lambda = trial.suggest_float('reg_lambda', 0.414, 0.789)

    global random_state

    model_1st = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=num_leaves,
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          n_estimators=n_estimators,
                          objective = 'fair',
                          min_split_gain=min_split_gain,
                          min_child_weight=min_child_weight,
                          min_child_samples=min_child_samples,
                          subsample=subsample,
                          subsample_freq=subsample_freq,
                          colsample_bytree=colsample_bytree,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )
    model_1st.fit(X_train_1st, np.log1p(y_train_1st))
    # 検証用データセットに対する予測
    y_pred = np.expm1(model_1st.predict(X_valid_1st))
    # 負の数値を0に変換
    y_pred = [0 if val <= 0 else val for val in y_pred]
    # RMSLEで評価
    from sklearn.metrics import root_mean_squared_log_error
    rmsle_score = root_mean_squared_log_error(y_valid_1st, y_pred)
    print(f"RMSLE: {rmsle_score:.4f}")
    return rmsle_score

study = optuna.create_study(direction='minimize')
study.optimize(run1, n_trials=100, show_progress_bar=True)
best_params = study.best_params
print('best_params:\t', best_params,'\nbest_value:\t',study.best_value)

model_1st = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=best_params['num_leaves'],
                          max_depth=best_params['max_depth'],
                          learning_rate=best_params['learning_rate'],
                          n_estimators=best_params['n_estimators'],
                          objective = 'fair',
                          min_split_gain=best_params['min_split_gain'],
                          min_child_weight=best_params['min_child_weight'],
                          min_child_samples=best_params['min_child_samples'],
                          subsample=best_params['subsample'],
                          subsample_freq=best_params['subsample_freq'],
                          colsample_bytree=best_params['colsample_bytree'],
                          reg_alpha=best_params['reg_alpha'],
                          reg_lambda=best_params['reg_lambda'],
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )

scores = []
# クロスバリデーションループ
for train_index, valid_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # トレーニングデータに対して対数変換を適用
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # モデルをトレーニングデータで学習
    model_1st.fit(X_train, y_train_log)

    # 検証用データセットに対する予測
    y_pred_log = model_1st.predict(X_valid)
    y_pred = np.expm1(y_pred_log)  # ログ変換を逆にして予測結果を元のスケールに戻す

    # 評価指標の計算
    cv_score = root_mean_squared_log_error(y_valid, y_pred)
    print(f"RMSLE: {cv_score}")
    scores.append(cv_score)
print("Mean RMSLE:", np.mean(scores))

from re import sub
from lightgbm import LGBMRegressor
def run2(trial):
    num_leaves = trial.suggest_int('num_leaves', 123, 190)
    max_depth = trial.suggest_int('max_depth', 114, 137)
    learning_rate = trial.suggest_float('learning_rate', 0.058, 0.24)
    n_estimators = trial.suggest_int('n_estimators', 532, 790)
    min_child_samples = trial.suggest_int('min_child_samples', 52, 74)
    min_split_gain = trial.suggest_float('min_split_gain', 0.867, 0.893)
    min_child_weight = trial.suggest_int('min_child_weight', 43, 73)
    subsample = trial.suggest_float('subsample', 0.587, 0.749)
    subsample_freq = trial.suggest_int('subsample_freq', 4, 6)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.528, 0.642)
    reg_alpha = trial.suggest_float('reg_alpha', 0.6195, 0.72)
    reg_lambda = trial.suggest_float('reg_lambda', 0.461, 0.531)
    global random_state

    model_2nd = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=num_leaves,
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          n_estimators=n_estimators,
                          objective='huber',
                          min_split_gain=min_split_gain,
                          min_child_weight=min_child_weight,
                          min_child_samples=min_child_samples,
                          subsample=subsample,
                          subsample_freq=subsample_freq,
                          colsample_bytree=colsample_bytree,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )
    model_2nd.fit(X_train_1st, np.log1p(y_train_1st))
    # 検証用データセットに対する予測
    y_pred = np.expm1(model_2nd.predict(X_valid_1st))
    # 負の数値を0に変換
    y_pred = [0 if val <= 0 else val for val in y_pred]
    # RMSLEで評価
    from sklearn.metrics import root_mean_squared_log_error
    rmsle_score = root_mean_squared_log_error(y_valid_1st, y_pred)
    print(f"RMSLE: {rmsle_score:.4f}")
    return rmsle_score

study = optuna.create_study(direction='minimize')
study.optimize(run2, n_trials=100, show_progress_bar=True)
best_params = study.best_params
print('best_params:\t', best_params,'\nbest_value:\t',study.best_value)

model_2nd = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=best_params['num_leaves'],
                          max_depth=best_params['max_depth'],
                          learning_rate=best_params['learning_rate'],
                          n_estimators=best_params['n_estimators'],
                          objective = 'huber',
                          min_split_gain=best_params['min_split_gain'],
                          min_child_weight=best_params['min_child_weight'],
                          min_child_samples=best_params['min_child_samples'],
                          subsample=best_params['subsample'],
                          subsample_freq=best_params['subsample_freq'],
                          colsample_bytree=best_params['colsample_bytree'],
                          reg_alpha=best_params['reg_alpha'],
                          reg_lambda=best_params['reg_lambda'],
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )
scores = []
# クロスバリデーションループ
for train_index, valid_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # トレーニングデータに対して対数変換を適用
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # モデルをトレーニングデータで学習
    model_2nd.fit(X_train, y_train_log)

    # 検証用データセットに対する予測
    y_pred_log = model_2nd.predict(X_valid)
    y_pred = np.expm1(y_pred_log)  # ログ変換を逆にして予測結果を元のスケールに戻す

    # 評価指標の計算
    cv_score = root_mean_squared_log_error(y_valid, y_pred)
    print(f"RMSLE: {cv_score}")
    scores.append(cv_score)
print("Mean RMSLE:", np.mean(scores))

from re import sub
from lightgbm import LGBMRegressor
def run3(trial):
    num_leaves = trial.suggest_int('num_leaves', 156, 197)
    max_depth = trial.suggest_int('max_depth', 152, 223)
    learning_rate = trial.suggest_float('learning_rate', 0.049, 0.0645)
    n_estimators = trial.suggest_int('n_estimators', 509, 652)
    min_child_samples = trial.suggest_int('min_child_samples', 50, 76)
    min_split_gain = trial.suggest_float('min_split_gain', 0.017, 0.02)
    min_child_weight = trial.suggest_int('min_child_weight', 28, 37)
    subsample = trial.suggest_float('subsample', 0.555, 0.608)
    subsample_freq = trial.suggest_int('subsample_freq', 3, 4)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.609, 0.711)
    reg_alpha = trial.suggest_float('reg_alpha', 0.522, 0.631)
    reg_lambda = trial.suggest_float('reg_lambda', 0.255, 0.603)
    global random_state

    model_3rd = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=num_leaves,
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          n_estimators=n_estimators,
                          objective='gamma',
                          min_split_gain=min_split_gain,
                          min_child_weight=min_child_weight,
                          min_child_samples=min_child_samples,
                          subsample=subsample,
                          subsample_freq=subsample_freq,
                          colsample_bytree=colsample_bytree,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )
    model_3rd.fit(X_train_1st, np.log1p(y_train_1st))
    # 検証用データセットに対する予測
    y_pred = np.expm1(model_3rd.predict(X_valid_1st))
    # 負の数値を0に変換
    y_pred = [0 if val <= 0 else val for val in y_pred]
    # RMSLEで評価
    from sklearn.metrics import root_mean_squared_log_error
    rmsle_score = root_mean_squared_log_error(y_valid_1st, y_pred)
    print(f"RMSLE: {rmsle_score:.4f}")
    return rmsle_score

study = optuna.create_study(direction='minimize')
study.optimize(run3, n_trials=100, show_progress_bar=True)
best_params = study.best_params
print('best_params:\t', best_params,'\nbest_value:\t',study.best_value)

model_3rd = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=best_params['num_leaves'],
                          max_depth=best_params['max_depth'],
                          learning_rate=best_params['learning_rate'],
                          n_estimators=best_params['n_estimators'],
                          objective = 'gamma',
                          min_split_gain=best_params['min_split_gain'],
                          min_child_weight=best_params['min_child_weight'],
                          min_child_samples=best_params['min_child_samples'],
                          subsample=best_params['subsample'],
                          subsample_freq=best_params['subsample_freq'],
                          colsample_bytree=best_params['colsample_bytree'],
                          reg_alpha=best_params['reg_alpha'],
                          reg_lambda=best_params['reg_lambda'],
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )
scores = []
# クロスバリデーションループ
for train_index, valid_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # トレーニングデータに対して対数変換を適用
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # モデルをトレーニングデータで学習
    model_3rd.fit(X_train, y_train_log)

    # 検証用データセットに対する予測
    y_pred_log = model_3rd.predict(X_valid)
    y_pred = np.expm1(y_pred_log)  # ログ変換を逆にして予測結果を元のスケールに戻す

    # 評価指標の計算
    cv_score = root_mean_squared_log_error(y_valid, y_pred)
    print(f"RMSLE: {cv_score}")
    scores.append(cv_score)
print("Mean RMSLE:", np.mean(scores))

from re import sub
from lightgbm import LGBMRegressor
def run4(trial):
    num_leaves = trial.suggest_int('num_leaves', 85, 119)
    max_depth = trial.suggest_int('max_depth', 146, 181)
    learning_rate = trial.suggest_float('learning_rate', 0.039, 0.0595)
    n_estimators = trial.suggest_int('n_estimators', 660, 766)
    min_child_samples = trial.suggest_int('min_child_samples', 17, 62)
    min_split_gain = trial.suggest_float('min_split_gain', 0.062, 0.078)
    min_child_weight = trial.suggest_int('min_child_weight', 78, 87)
    subsample = trial.suggest_float('subsample', 0.426, 0.521)
    subsample_freq = trial.suggest_int('subsample_freq', 3, 6)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.605, 0.71)
    reg_alpha = trial.suggest_float('reg_alpha', 0.45, 0.53)
    reg_lambda = trial.suggest_float('reg_lambda', 0.343, 0.563)
    global random_state

    model_4 = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=num_leaves,
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          n_estimators=n_estimators,
                          objective='tweedie',
                          min_split_gain=min_split_gain,
                          min_child_weight=min_child_weight,
                          min_child_samples=min_child_samples,
                          subsample=subsample,
                          subsample_freq=subsample_freq,
                          colsample_bytree=colsample_bytree,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )
    model_4.fit(X_train_1st, np.log1p(y_train_1st))
    # 検証用データセットに対する予測
    y_pred = np.expm1(model_4.predict(X_valid_1st))
    # 負の数値を0に変換
    y_pred = [0 if val <= 0 else val for val in y_pred]
    # RMSLEで評価
    from sklearn.metrics import root_mean_squared_log_error
    rmsle_score = root_mean_squared_log_error(y_valid_1st, y_pred)
    print(f"RMSLE: {rmsle_score:.4f}")
    return rmsle_score

study = optuna.create_study(direction='minimize')
study.optimize(run4, n_trials=100, show_progress_bar=True)
best_params = study.best_params
print('best_params:\t', best_params,'\nbest_value:\t',study.best_value)

model_4 = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=best_params['num_leaves'],
                          max_depth=best_params['max_depth'],
                          learning_rate=best_params['learning_rate'],
                          n_estimators=best_params['n_estimators'],
                          objective = 'tweedie',
                          min_split_gain=best_params['min_split_gain'],
                          min_child_weight=best_params['min_child_weight'],
                          min_child_samples=best_params['min_child_samples'],
                          subsample=best_params['subsample'],
                          subsample_freq=best_params['subsample_freq'],
                          colsample_bytree=best_params['colsample_bytree'],
                          reg_alpha=best_params['reg_alpha'],
                          reg_lambda=best_params['reg_lambda'],
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )

scores =[]

# クロスバリデーションループ
for train_index, valid_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # トレーニングデータに対して対数変換を適用
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # モデルをトレーニングデータで学習
    model_4.fit(X_train, y_train_log)

    # 検証用データセットに対する予測
    y_pred_log = model_4.predict(X_valid)
    y_pred = np.expm1(y_pred_log)  # ログ変換を逆にして予測結果を元のスケールに戻す

    # 評価指標の計算
    cv_score = root_mean_squared_log_error(y_valid, y_pred)
    print(f"RMSLE: {cv_score}")
    scores.append(cv_score)
print("Mean RMSLE:", np.mean(scores))

# クロスバリデーションループ
scores = []
for train_index, valid_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # トレーニングデータに対して対数変換を適用
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # モデルをトレーニングデータで学習
    model_1st.fit(X_train, y_train_log)
    model_2nd.fit(X_train, y_train_log)
    model_3rd.fit(X_train, y_train_log)
    model_4.fit(X_train, y_train_log)
    # 検証用データセットに対する予測
    y_pred_log = model_1st.predict(X_valid)
    y_pred_log2 = model_2nd.predict(X_valid)
    y_pred_log3 = model_3rd.predict(X_valid)
    y_pred_log4 = model_4.predict(X_valid)

    # 検証用データセットに対する予測
    y_pred_log = np.expm1((y_pred_log + y_pred_log2 + y_pred_log3 + y_pred_log4)/4)

    # 評価指標の計算
    cv_score = root_mean_squared_log_error(y_valid, y_pred_log)
    print(f"RMSLE: {cv_score}")
    scores.append(cv_score)
print("Mean RMSLE:", np.mean(scores))

model_1st.fit(X, np.log1p(y))
model_2nd.fit(X, np.log1p(y))
model_3rd.fit(X, np.log1p(y))
model_4.fit(X, np.log1p(y))
test_1st_1 = np.expm1(model_1st.predict(test_1st))
test_1st_2 = np.expm1(model_2nd.predict(test_1st))
test_1st_3 = np.expm1(model_3rd.predict(test_1st))
test_1st_4 = np.expm1(model_4.predict(test_1st))
test_pred = (test_1st_1 + test_1st_2 + test_1st_3 + test_1st_4) / 4

test_1st_1 = np.expm1(model_1st.predict(test_1st))
test_1st_2 = np.expm1(model_2nd.predict(test_1st))
test_1st_3 = np.expm1(model_3rd.predict(test_1st))
test_1st_4 = np.expm1(model_4.predict(test_1st))

# 出力データの形を結合できる形に直す。
test_2nd_1 = test_1st_1.reshape(-1, 1)
test_2nd_2 = test_1st_2.reshape(-1, 1)
test_2nd_3 = test_1st_3.reshape(-1, 1)
test_2nd_4 = test_1st_4.reshape(-1, 1)
# 出力データの結合と元の特徴量との結合
test_2nd = np.concatenate([test_2nd_1,test_2nd_2,test_2nd_3,test_2nd_4], axis=1)
test_2nd = pd.DataFrame(test_2nd, columns=['1st','2nd','3rd','4th'])

# 出力データ(メタラベル)の作成
X_2_1st = np.expm1(model_1st.predict(X))
X_2_2nd = np.expm1(model_2nd.predict(X))
X_2_3rd = np.expm1(model_3rd.predict(X))
X_2_4th = np.expm1(model_4.predict(X))
# 出力データの形を結合できる形に直す。
X_2_1st = X_2_1st.reshape(-1, 1)
X_2_2nd = X_2_2nd.reshape(-1, 1)
X_2_3rd = X_2_3rd.reshape(-1, 1)
X_2_4th = X_2_4th.reshape(-1, 1)
# 出力データの結合と元の特徴量との結合
X_2 = np.concatenate([X_2_1st,X_2_2nd,X_2_3rd,X_2_4th], axis=1)
X_2 = pd.DataFrame(X_2, columns=['1st','2nd','3rd','4th'])

test_X_2_diff_1st = test_2nd['1st'].median() / X_2['1st'].median()
test_X_2_diff_2nd = test_2nd['2nd'].median() / X_2['2nd'].median()
test_X_2_diff_3rd = test_2nd['3rd'].median() / X_2['3rd'].median()
test_X_2_diff_4th = test_2nd['4th'].median() / X_2['4th'].median()

X_2['1st'] = X_2['1st'] * test_X_2_diff_1st
X_2['2nd'] = X_2['2nd'] * test_X_2_diff_2nd
X_2['3rd'] = X_2['3rd'] * test_X_2_diff_3rd
X_2['4th'] = X_2['4th'] * test_X_2_diff_4th

X_train_2nd, X_valid_2nd= train_test_split(X_2, test_size=0.1, shuffle = True, random_state=random_state)

from re import sub
from lightgbm import LGBMRegressor
def run5(trial):
    num_leaves = trial.suggest_int('num_leaves', 121, 186)
    max_depth = trial.suggest_int('max_depth', 70, 110)
    learning_rate = trial.suggest_float('learning_rate', 0.03, 0.052)
    n_estimators = trial.suggest_int('n_estimators', 131, 177)
    min_child_samples = trial.suggest_int('min_child_samples', 64, 77)
    min_split_gain = trial.suggest_float('min_split_gain', 0.62, 0.84)
    min_child_weight = trial.suggest_int('min_child_weight', 47, 55)
    subsample = trial.suggest_float('subsample', 0.36, 0.43)
    subsample_freq = trial.suggest_int('subsample_freq', 4, 6)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.617, 0.644)
    reg_alpha = trial.suggest_float('reg_alpha', 0.556, 0.704)
    reg_lambda = trial.suggest_float('reg_lambda', 0.459, 0.653)
    global random_state

    model_5 = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=num_leaves,
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          n_estimators=n_estimators,
                          objective='mean_squared_error',
                          min_split_gain=min_split_gain,
                          min_child_weight=min_child_weight,
                          min_child_samples=min_child_samples,
                          subsample=subsample,
                          subsample_freq=subsample_freq,
                          colsample_bytree=colsample_bytree,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )
    model_5.fit(X_train_2nd, np.log1p(y_train_1st))
    # 検証用データセットに対する予測
    y_pred = np.expm1(model_5.predict(X_valid_2nd))
    # 負の数値を0に変換
    y_pred = [0 if val <= 0 else val for val in y_pred]
    # RMSLEで評価
    from sklearn.metrics import root_mean_squared_log_error
    rmsle_score = root_mean_squared_log_error(y_valid_1st, y_pred)
    print(f"RMSLE: {rmsle_score:.4f}")
    return rmsle_score

study = optuna.create_study(direction='minimize')
study.optimize(run5, n_trials=100, show_progress_bar=True)
best_params = study.best_params
print('best_params:\t', best_params,'\nbest_value:\t',study.best_value)

model_5 = LGBMRegressor(boosting_type='gbdt',
                          num_leaves=best_params['num_leaves'],
                          max_depth=best_params['max_depth'],
                          learning_rate=best_params['learning_rate'],
                          n_estimators=best_params['n_estimators'],
                          objective = 'mean_squared_error',
                          min_split_gain=best_params['min_split_gain'],
                          min_child_weight=best_params['min_child_weight'],
                          min_child_samples=best_params['min_child_samples'],
                          subsample=best_params['subsample'],
                          subsample_freq=best_params['subsample_freq'],
                          colsample_bytree=best_params['colsample_bytree'],
                          reg_alpha=best_params['reg_alpha'],
                          reg_lambda=best_params['reg_lambda'],
                          random_state=random_state,
                          n_jobs=-1,
                          verbosity=-1,

    )

scores =[]

# クロスバリデーションループ
for train_index, valid_index in kf.split(X_2):
    X_train, X_valid = X_2.iloc[train_index], X_2.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # トレーニングデータに対して対数変換を適用
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # モデルをトレーニングデータで学習
    model_5.fit(X_train, y_train_log)

    # 検証用データセットに対する予測
    y_pred_log = model_5.predict(X_valid)
    y_pred = np.expm1(y_pred_log)  # ログ変換を逆にして予測結果を元のスケールに戻す

    # 評価指標の計算
    cv_score = root_mean_squared_log_error(y_valid, y_pred)
    print(f"RMSLE: {cv_score}")
    scores.append(cv_score)
print("Mean RMSLE:", np.mean(scores))

model_5.fit(X_2, np.log1p(y))
test_pred = np.expm1(model_5.predict(test_2nd))

# 投稿ファイル作成
submit = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/SMBC(Carbon)/data/sample_submission.csv', header=None)
submit[1] = test_pred
submit.to_csv('/content/drive/MyDrive/Colab Notebooks/SMBC(Carbon)/data/sample_submission.csv', header=None, index=False)
