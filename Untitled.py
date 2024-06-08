# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np

# 예제 데이터 생성
data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
# -

from sklearn.preprocessing import MinMaxScaler

# +
# Min-Max 스케일러 객체 생성
scaler = MinMaxScaler()

# 데이터프레임 정규화
df_minmax = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("\nMin-Max Normalized DataFrame:")
print(df_minmax)
# -


