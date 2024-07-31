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
# fit: 데이터의 최소값과 최대값을 계산, 이 단계에서 스케일러는 데이터의 통계를 학습
# transform: 데이터를 0과 1사이의 값으로 변환. 이 값은 'fit'에서 학습된 최소값과 최대값을 사용
# scaler.fit_transform(df)는 numpy 배열을 반환. 이 배열을 pandas DataFrame으로 변환하기 위해 pd.DataFrame 생성자 사용
# columns=df.colunms: 원본 데이터 프레임 'df'의 열 이름을 사용하여 새로운 데이터 프레임의 열 이름을 동일하게 유지
print("\nMin-Max Normalized DataFrame:")
print(df_minmax)
# -

from sklearn.preprocessing import StandardScaler

# +
# Standard 스케일러 객체 생성
scaler = StandardScaler()

# 데이터프레임 정규화
df_standard = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# fit: 각 피처의 평균과 표준편차를 계산, 이 단계에서 스케일러는 데이터의 통계를 학습
# transform: 데이터의 평균을 0, 표준편차를 1. 이 값은 'fit'에서 학습된 평균과 표준편차를 사용
print("\nZ-score Normalized DataFrame:")
print(df_standard)
# -


