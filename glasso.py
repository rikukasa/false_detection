import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV

# データの読み込み
df = pd.read_csv('watertreatment_mod.csv', encoding='shift_jis', header=0, index_col=0)
TimeIndices = df.index

# 訓練データとテストデータの分割
split_point = 100
offset = 0

# 複数の訓練データセットを作成 (例として3つのセット)
train_dfs = [
    df.iloc[offset:offset+split_point,],
    df.iloc[offset+split_point:offset+2*split_point,],
    df.iloc[offset+2*split_point:offset+3*split_point,]
]

test_df = df.iloc[offset+split_point*3:offset+split_point*4,]  # テストデータの選択

# リプレイ攻撃をシミュレートするために、テストデータの一部を訓練データの値で置き換える
num = 20  # 攻撃を行う列の指定

# # 訓練データとテストデータの長さを確認して、短い方に合わせてスライス
min_len = min(len(test_df), len(train_dfs[0]))
test_df.iloc[:min_len, num] = train_dfs[0].iloc[:min_len, num]  # リプレイ攻撃をシミュレーション

print(test_df.head())  # 置き換え後のテストデータを確認

# 標準化関数
def standardize(_df):
    sc = StandardScaler()
    sc.fit(_df)
    df_std = sc.transform(_df)
    return df_std

# Graphical Lassoを用いて共分散行列と精度行列を推定
def glasso_cov_prec_(array):
    model = GraphicalLassoCV()
    model.fit(array)
    cov_ = model.covariance_
    prec_ = model.precision_
    return cov_, prec_

# 複数の訓練データセットの共分散行列と精度行列を取得
train_covs = []
train_precs = []

for train_df in train_dfs:
    train_df_std = standardize(train_df)
    train_cov_, train_prec_ = glasso_cov_prec_(train_df_std)
    train_covs.append(train_cov_)
    train_precs.append(train_prec_)

# テストデータの標準化と共分散行列、精度行列の取得
test_df_std = standardize(test_df)
test_cov_, test_prec_ = glasso_cov_prec_(test_df_std)

# 複数の訓練データセットに対して異常度 a を算出
def compute_anomaly_scores(train_cov, train_prec, test_cov, test_prec):
    a = [0] * test_cov.shape[0]
    b1 = np.matmul(np.matmul(train_prec, train_cov), train_prec)
    b2 = np.matmul(np.matmul(test_prec, train_cov), test_prec)
    c1 = np.matmul(np.matmul(train_prec, test_cov), train_prec)
    c2 = np.matmul(np.matmul(test_prec, test_cov), test_prec)
    
    for i in range(test_cov.shape[0]):
        v1 = 1 / 2 * np.log(train_cov[i, i] / test_cov[i, i]) - 1 / 2 * (b1[i, i] / train_prec[i, i] - b2[i, i] / test_prec[i, i])
        v2 = 1 / 2 * np.log(test_cov[i, i] / train_cov[i, i]) - 1 / 2 * (c2[i, i] / test_prec[i, i] - c1[i, i] / train_prec[i, i])
        a[i] = max(v1, v2)
    
    return a

# 複数の訓練データセットに対して異常度を計算
anomaly_scores_all = []
for train_cov_, train_prec_ in zip(train_covs, train_precs):
    anomaly_scores = compute_anomaly_scores(train_cov_, train_prec_, test_cov_, test_prec_)
    anomaly_scores_all.append(anomaly_scores)
    
# 異常度の可視化（縦棒グラフ）
plt.figure(figsize=(10, 6))

# 複数の訓練データセットに対して異常度を縦棒グラフでプロット
for i, anomaly_scores in enumerate(anomaly_scores_all):
    plt.bar(np.arange(len(anomaly_scores)) + i * 0.2, anomaly_scores, width=0.2, label=f'Training Set {i+1}')

plt.title('Anomaly Scores for Multiple Training Sets (Bar Plot)')
plt.xlabel('Feature Index')
plt.ylabel('Anomaly Score')
plt.legend()
plt.xticks(np.arange(len(anomaly_scores)))  # x軸の目盛りを特徴のインデックスに合わせる
plt.show()

# 異常度の値を表示
for i, anomaly_scores in enumerate(anomaly_scores_all):
    print(f"Anomaly Scores for Training Set {i+1}:")
    print(anomaly_scores)