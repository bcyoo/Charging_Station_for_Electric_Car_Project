# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# 필요 라이브러리 
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as plt
import seaborn as sns

# # 데이터 불러오기

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 데이터 read
df_1 = gpd.read_file("17.고성_건물분포도(연면적)_격자.json", encoding='euc-kr')
df_2 = gpd.read_file("17.속초_건물분포도(연면적)_격자.json", encoding='euc-kr')


# data 확인
df_1

df_2

# 강원도_토지소유(1)~(3).geojson을 concat으로 행방향 병합
# ignore_index=True로 index 재배열
df_test = pd.concat([df_1, df_2], ignore_index=True)

# 확인
df_test

# data 저장
df_test.to_file("속초-고성_건물분포도(연면적)_격자.json", encoding='utf-8')















# # 데이터 전처리

# ## column 정의서를 이용한 column명 변경

# column 확인
df_1.columns, df_2.columns, df_3.columns

# column 이름 변경
df_1.columns = ['고유번호', '법정동코드', '법정동명', '대장구분코드', '대장구분명', '지번', '지번지목부호', '소유구분코드',
'소유구분명', '공유인수', '연령대구분코드', '연령대구분', '거주지구분코드', '거주지구분', '국가기관구분코드', '국가기관구분',
 '소유권변동원인코드', '소유권변동원인', '소유권변동일자', '지목코드', '지목', '라벨', '토지면적', '데이터기준일자', 'geometry']
df_2.columns = ['고유번호', '법정동코드', '법정동명', '대장구분코드', '대장구분명', '지번', '지번지목부호', '소유구분코드',
'소유구분명', '공유인수', '연령대구분코드', '연령대구분', '거주지구분코드', '거주지구분', '국가기관구분코드', '국가기관구분',
 '소유권변동원인코드', '소유권변동원인', '소유권변동일자', '지목코드', '지목', '라벨', '토지면적', '데이터기준일자', 'geometry']
df_3.columns = ['고유번호', '법정동코드', '법정동명', '대장구분코드', '대장구분명', '지번', '지번지목부호', '소유구분코드',
'소유구분명', '공유인수', '연령대구분코드', '연령대구분', '거주지구분코드', '거주지구분', '국가기관구분코드', '국가기관구분',
 '소유권변동원인코드', '소유권변동원인', '소유권변동일자', '지목코드', '지목', '라벨', '토지면적', '데이터기준일자', 'geometry']

# column 확인
df_1.columns, df_2.columns, df_3.columns

# ## 데이터 인덱싱(강원도 -> 속초, 고성)

# "속초시" data 개수
df_1["법정동명"].str.contains("속초시").sum()

# "고성군" data 개수
df_1["법정동명"].str.contains("고성군").sum()

# "속초시" data, "고성군" data 합계
(df_1["법정동명"].str.contains("속초시") | df_1["법정동명"].str.contains("고성군")).sum()

# "속초시", "고성군" data
df_1 = df_1[df_1["법정동명"].str.contains("속초시") | df_1["법정동명"].str.contains("고성군")]
df_2 = df_2[df_2["법정동명"].str.contains("속초시") | df_2["법정동명"].str.contains("고성군")]
df_3 = df_3[df_3["법정동명"].str.contains("속초시") | df_3["법정동명"].str.contains("고성군")]

# 데이터 확인
df_1.head()

# 각 데이터 수
len(df_1), len(df_2), len(df_3)

# -  강원도_토지소유(1)~(3).geojson에 일괄적용

# ## 불필요한 columns 제거

# reference(광양시) 확인
df = gpd.read_file('reference data/14.광양시_소유지정보.geojson', encoding='cp949')
df.head()

# reference column 확인
df.columns

# column 확인
df_1.columns

# 필요없는 column 제거
df_1.drop(['대장구분코드', '대장구분명', '지번지목부호', '공유인수', '연령대구분코드', '연령대구분', '거주지구분코드', 
         '거주지구분', '소유권변동원인코드', '소유권변동원인', '소유권변동일자','라벨','데이터기준일자'], axis=1, inplace=True)
df_2.drop(['대장구분코드', '대장구분명', '지번지목부호', '공유인수', '연령대구분코드', '연령대구분', '거주지구분코드', 
         '거주지구분', '소유권변동원인코드', '소유권변동원인', '소유권변동일자','라벨','데이터기준일자'], axis=1, inplace=True)
df_3.drop(['대장구분코드', '대장구분명', '지번지목부호', '공유인수', '연령대구분코드', '연령대구분', '거주지구분코드', 
         '거주지구분', '소유권변동원인코드', '소유권변동원인', '소유권변동일자','라벨','데이터기준일자'], axis=1, inplace=True)

# drop 확인
df_1

# ## 토지소유(1)~(3).geojson 병합

# 강원도_토지소유(1)~(3).geojson을 concat으로 행방향 병합
# ignore_index=True로 index 재배열
df_test = pd.concat([df_1, df_2], ignore_index=True)

# 확인
df_test

# ## 결측치 처리

# data의 전체 개수 확인
len(df_test)

# 결측치 개수 확인
df_test.isna().sum()

# 결측치를 제거
df_test.dropna(axis=0, inplace=True)

# index_reset
df_test.reset_index(drop=True, inplace=True)

# 확인
df_test

# ## 데이터 전처리 - 데이터 수정 및 변경

# 소유구분명에 이상값을 확인
df_test.소유구분명.unique()

# '소유구분명' == ""의 경우 소유자구분이 안되는 것을 확인
df_test[df_test['소유구분명'] == ""]

# 결측치는 가졌던 데이터의 1건은 '소유구분코드' == "ZZ"임을 확인
df_test[df_test['소유구분코드'] == "ZZ"]

# 결측치 확인
df_test.isna().sum()

# # 데이터 저장

# data 저장
df_test.to_file("속초-고성_법정동별_인구현황.json", encoding='utf-8')
