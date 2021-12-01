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

# +
import pathlib
import random
from functools import reduce
from collections import defaultdict

import pandas as pd
import geopandas as gpd # 설치가 조금 힘듭니다. 어려우시면 https://blog.naver.com/PostView.nhn?blogId=kokoyou7620&logNo=222175705733 참고하시기 바랍니다.
import folium
import shapely 
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import sklearn.cluster
import tensorflow as tf  # 설치 따로 필요합니다. https://chancoding.tistory.com/5 참고 하시면 편해요.

#from geoband import API         이건 설치 필요 없습니다.

import pydeck as pdk                  # 설치 따로 필요합니다.
import os

import pandas as pd


import cufflinks as cf                 # 설치 따로 필요합니다.   
cf.go_offline(connected=True)
cf.set_config_file(theme='polar')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Nanum Gothic'

import numpy as np
from shapely.geometry import Polygon, Point
from numpy import random

import geojson                       # 밑에 Line 84에 추가하여야 하지만 바로 import 안되서 설치 필요

#최적화 solver
import time
from mip import Model, xsum, maximize, BINARY  # 설치 따로 필요합니다.
# -

import sys
'geopandas' in sys.modules

# +
#Pydeck 사용을 위한 함수 정의
import geopandas as gpd 
import shapely # Shapely 형태의 데이터를 받아 내부 좌표들을 List안에 반환합니다. 
def line_string_to_coordinates(line_string): 
    if isinstance(line_string, shapely.geometry.linestring.LineString): 
        lon, lat = line_string.xy 
        return [[x, y] for x, y in zip(lon, lat)] 
    elif isinstance(line_string, shapely.geometry.multilinestring.MultiLineString): 
        ret = [] 
        for i in range(len(line_string)): 
            lon, lat = line_string[i].xy 
            for x, y in zip(lon, lat): 
                ret.append([x, y])
        return ret 

def multipolygon_to_coordinates(x): 
    lon, lat = x[0].exterior.xy 
    return [[x, y] for x, y in zip(lon, lat)] 

def polygon_to_coordinates(x): 
    lon, lat = x.exterior.xy 
    return [[x, y] for x, y in zip(lon, lat)] 



# -

def multipolygon_to_coordinates(series_row): 
    lon, lat = series_row[0].exterior.xy
    return [[x, y] for x, y in zip(lon, lat)]
def polygon_to_coordinates(series_row):
    lon, lat = series_row.exterior.xy
    return [[x, y] for x, y in zip(lon, lat)]
mp_idx = []
p_idx = []
for i in range(len(df_10)):
#     print(df_20['geometry'][i].geom_type)
    if df_10['geometry'][i].geom_type == 'MultiPolygon':
        mp_idx.append(i)
#         df_20['coordinates'].iloc[i] = multipolygon_to_coordinates(df_20['geometry'][i])
    if df_10['geometry'][i].geom_type == 'Polygon':
#         df_20['coordinates'] = polygon_to_coordinates(df_20['geometry'][i])
        p_idx.append(i)
# df_20.head()
df_10['coordinates'] = 0
for idx1 in p_idx:
    print(idx1)
#     if idx1 == df_20['geometry'].index.any():
    df_10['coordinates'].iloc[idx1] = polygon_to_coordinates(df_10['geometry'][idx1])
#     else:
#         pass
for idx2 in mp_idx:
#     if idx2 == df_20['geometry'].index.any():
#         df_20['coordinates'] = 0
        df_10['coordinates'].iloc[idx2] = multipolygon_to_coordinates(df_10['geometry'][idx2])
#     else:
#         pass
df_10.head()

df_10= gpd.read_file("MOCT_LINK.json")
df_10

df_10

df = df_10
df = df_10
df['coordinate'] = df['geometry'].buffer(0.001).apply(polygon_to_coordinates) 
df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다. 
df['정규화도로폭'] = df['width'].apply(int) / df['width'].apply(int).max()

df['정규화도로폭'] = df_15['WIDTH'].apply(int) / df_15['WIDTH'].apply(int).max()

df_15 = pd.read_csv('속초-고성_상세지도망.csv')
df_15.info()

df_15.astype({'WIDTH':int})

df_12 = pd.read_csv('속초-고성_2017_혼잡빈도,시간,기대_강도.csv')
df_12

# +
# 혼합빈도강도 양방향 총 합
df_10_ = []
for i in df_10.LINK_ID:
    df_10_.append([i,sum(df_13[df_13['LINK_ID'].apply(str).str.contains(i)].혼잡시간강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["LINK_ID", "혼잡시간강도합"]
df_10_13=pd.merge(df, df_10_,on = 'LINK_ID' )
# 혼잡시간강도 합이 가장 높은 도로
df_10_13.iloc[df_10_13["혼잡시간강도합"].sort_values(ascending=False).index].reindex().head()
# -

layer = pdk.Layer( 'PathLayer', 
                  df_10, 
                  get_path='geometry', 
                  get_width='혼잡빈도강도', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918488, 38.2070148] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state
            ) 
r.to_html()
