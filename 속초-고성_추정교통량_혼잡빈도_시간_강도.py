# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
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

#최적화 solver
import time
from mip import Model, xsum, maximize, BINARY  # 설치 따로 필요합니다.

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

# ---------------------------------
# # I. EDA
#
# ## 1. 인구현황 분석
#
#
# **목적: 격자 별 인구 현황을 확인**
#
# **분석 데이터 종류**
# - df_08: 격자별인구현황.csv"
#
# **분석 설명**
# - 초록색일 수록 인구가 많이 있으며, 검은색일 수록 인구가 적으며, 색이 칠해지지 않은 곳은 인구 현황 값이 0 이다.
# - 인구현황데이터는 현재 100X100 grid로 나누어져 있으나 추후 분석을 위해 grid의 중심에 해당하는 Point 값 (Central point)을 계산해 주었고, 각각에 고유한 grid id를 부여했다.
# - 따라서 인구 현황을 100X100 point로 설명 할 수 있는 결과를 도출하였다. 
#

df_08= gpd.read_file("08.속초-고성_격자별인구현황.json")
df_08

# +
# 격자별 인구 현황
df_08= gpd.read_file("08.속초-고성_격자별인구현황.json")

# val 열 na 제거
df_08['val'] = df_08['val'].fillna(0)


# 인구 수 정규화
df_08['정규화인구'] = df_08['val'] / df_08['val'].max()


# geometry를 coordinate 형태로 적용
df_08['coordinates'] = df_08['geometry'].apply(polygon_to_coordinates) #pydeck 을 위한 coordinate type

# 100X100 grid에서 central point 찾기
df_08_list = []
df_08_list2 = []
for i in df_08['geometry']:
    cent = [[i.centroid.coords[0][0],i.centroid.coords[0][1]]]
    df_08_list.append(cent)
    df_08_list2.append(Point(cent[0]))
df_08['coord_cent'] = 0
df_08['geo_cent'] = 0
df_08['coord_cent']= pd.DataFrame(df_08_list) # pydeck을 위한 coordinate type
df_08['geo_cent'] = df_08_list2 # geopandas를 위한 geometry type


# 쉬운 분석을 위한 임의의 grid id 부여
df_08['grid_id']=0
idx = []
for i in range(len(df_08)):
    idx.append(str(i).zfill(5))
df_08['grid_id'] = pd.DataFrame(idx)

# 인구 현황이 가장 높은 위치
df_08.iloc[df_08["val"].sort_values(ascending=False).index].reindex().head()

# +
# Make layer
# 사람이 있는 그리드만 추출
layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_08[(df_08['val'].isnull()==False) & df_08['val']!=0], # 시각화에 쓰일 데이터프레임 
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[900, 255*정규화인구, 0, 정규화인구*10000 ]', # 각 데이터 별 rgb 또는 rgba 값 (0~255)
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

# Set the viewport location 
center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
)

# Render 
r = pdk.Deck(layers=[layer], initial_view_state=view_state)

r.to_html()
# -

# ---------------------------------
# ## 4. 교통량 분석
#
#
# **목적: 주거시간과 업무시간의 교통량 분석**
#
# **분석 데이터 종류**
# - df_10: 상세도로망.geojson
# - df_11: 평일_일별_시간대별_추정교통량.csv"
#
# **분석그래프 설명**
# - 도로폭이 넓을 수록 노란색이며 좁을 수록 붉은색이다.
# - 선이 굵을 수록 혼잡빈도강도가 높은 것이며, 얇을 수록 낮은 것이다

df_11 = pd.read_csv('11.평일_일별_시간대별__추정교통량.csv')

df = df_10
df = df_10
df['coordinate'] = df['geometry'].buffer(0.001).apply(polygon_to_coordinates) 
df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다. 
df['정규화도로폭'] = df['width'].apply(int) / df['width'].apply(int).max()

# +

# 여행객들이 11시에 관광지로 움직일 것으로 가정
# 승용차만 고려
df_11_time11 = df_11[df_11['시간적범위']==11]

df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_11_time11[df_11_time11['link_id'].apply(str).str.contains(i)]['승용차'])])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "교통량"]
df_10_11_time11=pd.merge(df, df_10_,on = 'link_id' )

# 교통량 합이 가장 높은 도로
df_10_11_time11.iloc[df_10_11_time11["교통량"].sort_values(ascending=False).index].reindex().head()


# +
layer = pdk.Layer( 'PathLayer', 
                  df_10_11_time11, 
                  get_path='coordinate', 
                  get_width='교통량/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 



center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state
            ) 
r.to_html()

# -

# ### 분석 결과
#
# - 7시에 교통량이 많은 곳은 주거지역으로 간주하였으며, 중마동, 마동은 주거지역일 것으로 기대
#

# +
# 대부분의 사람은 오후 3시에 업무를 하는 것으로 가정 (운송 업 포함)
df_11_time17=df_11[df_11['시간적범위']==17]

df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_11_time17[df_11_time17['link_id'].apply(str).str.contains(i)]['승용차'])])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "교통량"]
df_10_11_time17=pd.merge(df, df_10_,on = 'link_id' )

# 교통량 합이 가장 높은 도로
df_10_11_time17.iloc[df_10_11_time17["교통량"].sort_values(ascending=False).index].reindex().head()
# -

layer = pdk.Layer( 'PathLayer', 
                  df_10_11_time17, 
                  get_path='coordinate', 
                  get_width='교통량/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state
            ) 
r.to_html()

# ### 분석 결과
# - 15시에 교통량이 많은 곳은 상업지역 혹은 차량이 가장 많이 지나는 중심 도로 간주하였으며,  광양읍은 업무 중심 지역일 것으로 기대
# - 급속 충전소는 업무지역 설치가 효과적이라 판단
#
# -------------------------------------
#
#

# ---------------------------------
# ## 5. 혼잡빈도강도, 혼잡시간강도 분석
#
#
# **목적: 혼잡빈도강도와 혼잡시간빈도를 분석하여 차량이 많은 위치 파악**
#
# **분석 데이터 종류**
# - df_10: 상세도로망
# - df_12: 평일_전일_혼잡빈도강도.csv"
# - df_13: 평일_전일_혼잡시간강도.csv"
#
# **분석그래프 설명**
# - 도로폭이 넓을 수록 노란색이며 좁을 수록 붉은색이다.
# - 선이 굵을 수록 혼잡빈도강도가 높은 것이며, 얇을 수록 낮은 것이다

df_10= gpd.read_file("강원도속초시,고성군_상세도로망.json")
# df_11= pd.read_csv("11.광양시_평일_일별_시간대별_추정교통량.csv")
df_12= pd.read_csv("12.평일_혼잡빈도강도_강원도 속초시, 고성군.csv")
df_13= pd.read_csv("13.평일_혼잡시간강도_강원도 속초시, 고성군.csv")

df = df_10
df = df_10
df['coordinate'] = df['geometry'].buffer(0.001).apply(polygon_to_coordinates) 
df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다. 
df['정규화도로폭'] = df['width'].apply(int) / df['width'].apply(int).max()

# +
# 혼합빈도강도 양방향 총 합
df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_12[df_12['link_id'].apply(str).str.contains(i)].혼잡빈도강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "혼잡빈도강도합"]
df_10_12=pd.merge(df, df_10_,on = 'link_id' )

# 혼잡빈도강도 합이 가장 높은 도로
df_10_12.iloc[df_10_12["혼잡빈도강도합"].sort_values(ascending=False).index].reindex().head()


# -

layer = pdk.Layer( 'PathLayer', 
                  df_10_12, 
                  get_path='coordinate', 
                  get_width='혼잡빈도강도합/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
#             mapbox_key = 'sk.eyJ1IjoieW9vYnl1bmdjaHVsIiwiYSI6ImNrd245YnMwZzFiMnEycHBkc2gzbzkzd3AifQ.sc9Gmo56AsAHzJ2B3wCkXg') 
r.to_html()

# +
# - 도로폭이 넓을 수록 노란색이며 좁을 수록 붉은색이다.
# - 선이 굵을 수록 혼잡빈도강도가 높은 것이며, 얇을 수록 낮은 것이다

# +
# 혼합시간강도 양방향 총 합
df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_13[df_13['link_id'].apply(str).str.contains(i)].혼잡시간강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "혼잡시간강도합"]
df_10_13=pd.merge(df, df_10_,on = 'link_id' )
# 혼잡시간강도 합이 가장 높은 도로
df_10_13.iloc[df_10_13["혼잡시간강도합"].sort_values(ascending=False).index].reindex().head()
# -

layer = pdk.Layer( 'PathLayer', 
                  df_10_13, 
                  get_path='coordinate', 
                  get_width='혼잡시간강도합/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
#             mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g") 
r.to_html()

# ---------------------------------
# ## 6. 급속충전소 설치가능 장소 필터링
#
#
# **목적: 급속 충전소의 경우 사유지는 제외 해야 하므로 설치 가능 장소 필터링 필요**
#
# **분석 데이터 종류**
# - df_14: 속초시_소유지정보.csv"
#
# **분석 설명**
# - 사유지를 포함한 임야, 염전, 도로, 철도 용지, 제방, 하천과 같이 설치가 부적절 한 곳을 필터링 한 multipolygone을 시각화하였다.
# - 앞서 도출한 인구현황 100X100 Point 데이터셋에서 설치가능한 장소에 해당하는 point를 추출하였다.
#

df_14= gpd.read_file("./14.소유지정보.geojson") # geojson -> json

# +
# 급속 충전소 설치 가능 장소
df_14_=df_14[df_14['소유구분코드'].isin(['02','04'])] #소유구분코드: 국유지, 시/군
df_14_possible=df_14[df_14['소유구분코드'].isin(['02','04']) 
      & (df_14['지목코드'].isin(['05','07','14','15','16','17',
                             '18','19','20','27' ])==False)] # 임야, 염전, 도로, 철도 용지, 제방, 하천 제외 

# geometry to coordinates
df_14_possible['coordinates'] = df_14_possible['geometry'].apply(polygon_to_coordinates) 

# 설치가능한 모든 polygone을 multipolygone으로 묶음
from shapely.ops import cascaded_union
boundary = gpd.GeoSeries(cascaded_union(df_14_possible['geometry'].buffer(0.001)))

from geojson import Feature, FeatureCollection, dump
MULTIPOLYGON = boundary[0]

features = []
features.append(Feature(geometry=MULTIPOLYGON, properties={"col": "privat"}))
feature_collection = FeatureCollection(features)
with open('geo_possible.geojson', 'w') as f:
   dump(feature_collection, f)

geo_possible= gpd.read_file("geo_possible.geojson")
# -

# 브로드캐스팅을 이용한 요소합 (평행이동)
# 요소합 진행 후, 마지막 데이터를 list로 형변환
v = np.array([-0.0022, 0.0027])
for i in range(len(df_14_possible["coordinates"])):
    for j in range(len(df_14_possible["coordinates"].iloc[i])):
                   df_14_possible["coordinates"].iloc[i][j] = list(df_14_possible["coordinates"].iloc[i][j] + v)
df_14_possible["coordinates"]

# +
layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_14_possible, # 시각화에 쓰일 데이터프레임
                  #df_result_fin[df_result_fin['val']!=0],
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[0, 255*1, 0,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

# Set the viewport location 
center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 


# Render 
r = pdk.Deck(layers=[layer], initial_view_state=view_state,
            ) 

    
r.to_html()
# -

# # II. 입지선정지수 개발
#
# ## 1. 지역특성 요소 추출
#
# ### 100X100 Point 중 설치 가능 한 Point 필터링 
# - 100X100 중 설치 가능한 multipolygone에 있는 point를 필터링하는 시간이 굉장히 오래 소요됨(약 1시간)
# - 따라서 이를 모두 계산한 'within_points.csv'를 따로 저장하여 첨부함
# - df_result로 최종 분석 할 데이터셋을 만듦

# +
# 최종 분석 데이터 정제하기

# 개발 가능 한 grid point 찾기
shapely.speedups.enable()
df_result = df_08[['grid_id','val','geometry','coordinates','coord_cent','geo_cent']]
df_result['val'] = df_result['val'].fillna(0)

# #굉장히 오래걸림
# point_cent= gpd.GeoDataFrame(df_result[['grid_id','geo_cent']],geometry = 'geo_cent')
# within_points=point_cent.buffer(0.00000001).within(geo_possible.loc[0,'geometry'])
# pd.DataFrame(within_points).to_csv("within_points.csv", index = False)

within_points=pd.read_csv("within_points.csv")
df_result['개발가능'] = 0
df_result['개발가능'][within_points['0']==True] = 1
df_result[df_result['개발가능']==1]
# -

# ## 2. 100X100 Point에 교통량, 혼잡빈도강도, 혼잡시간강도 관련 요소 부여
#
# **목적: grid 마다 교통량 관련 요소 부여**
#
# **분석 데이터 종류**
# - df_11: 평일_일별_시간대별_추정교통량.csv"
# - df_12: 평일_전일_혼잡빈도강도.csv"
# - df_13: 평일_전일_혼잡시간강도.csv"
#
# **분석 설명**
# - 각 100X100 Point 마다 07시 교통량, 15시 교통량, 혼잡빈도강도합, 혼잡시간강도합을 부여
# - 각 요소바다 부여하는데 시간이 다소 소요됨 (약 10분)

# +
# grid 마다 11시 교통량 부여
df_10_11_time11_grid = []
df_superset = df_10_11_time11[df_10_11_time11['교통량']>0]

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_10_11_time11_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_10_11_time11_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_10_11_time11_grid)):
    id_idx = df_10_11_time11_grid[i][0]
    grids = df_10_11_time11_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['교통량'])])
    except:
        pass

#2시 승용차 혼잡 빈도 관련 정보
try:
    del df_result['교통량_11']
except:
    pass

grid_=pd.DataFrame(grid_list)
grid_.columns = ["grid_id","교통량_11"]
#grid_혼잡빈도[grid_혼잡빈도['승용차_혼잡빈도강도합']>0]
df_result = pd.merge(df_result, grid_, on = 'grid_id')
df_result[df_result['교통량_11']>0]

# +
# grid 마다 17시 교통량 부여
df_10_11_time17_grid = []
df_superset = df_10_11_time17[df_10_11_time17['교통량']>0]

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_10_11_time17_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_10_11_time17_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_10_11_time17_grid)):
    id_idx = df_10_11_time17_grid[i][0]
    grids = df_10_11_time17_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_혼잡빈도_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_혼잡빈도_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['교통량'])])
    except:
        pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","교통량_17"]
grid_혼잡빈도[grid_혼잡빈도['교통량_17']>0]


#17시 승용차 혼잡 빈도 관련 정보
try:
    del df_result['교통량_17']
except:
    pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","교통량_17"]
#grid_혼잡빈도[grid_혼잡빈도['승용차_혼잡빈도강도합']>0]
df_result = pd.merge(df_result, grid_혼잡빈도, on = 'grid_id')
df_result[df_result['교통량_17']>0]

# +
# grid 마다 혼잡빈도강도 부여
df_10_grid = []
df_superset = df_10_12[df_10_12['혼잡빈도강도합']>0]

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_10_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_10_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_10_grid)):
    id_idx = df_10_grid[i][0]
    grids = df_10_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_혼잡빈도_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_혼잡빈도_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['혼잡빈도강도합'])])
    except:
        pass

#혼잡빈도강도 관련 정보
try:
    del df_result['혼잡빈도강도합']
except:
    pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","혼잡빈도강도합"]
#grid_혼잡빈도[grid_혼잡빈도['승용차_혼잡빈도강도합']>0]
df_result = pd.merge(df_result, grid_혼잡빈도, on = 'grid_id')
df_result[df_result['혼잡빈도강도합']>0]

# +
# grid 마다 혼잡시간강도합 부여
df_10_grid = []
df_superset = df_10_13[df_10_13['혼잡시간강도합']>0]

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_10_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_10_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_10_grid)):
    id_idx = df_10_grid[i][0]
    grids = df_10_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_혼잡빈도_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_혼잡빈도_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['혼잡시간강도합'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['혼잡시간강도합']
except:
    pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","혼잡시간강도합"]
#grid_혼잡빈도[grid_혼잡빈도['승용차_혼잡빈도강도합']>0]
df_result = pd.merge(df_result, grid_혼잡빈도, on = 'grid_id')
df_result[df_result['혼잡시간강도합']>0]
# -

# ---------------------------------
# ## 5. 기존 충전소 위치 분석
#
#
# **목적: 기존 충전소가 있는 위치를 분석, 기존 충전소가 커버가능한 범위는 제외하고 분석**
#
# **분석 데이터 종류**
# - df_01: 01.고성군_속초시_충전기설치현황.csv
#
# **분석 설명**
# - 급속 충전소 (Fast-charing Station, FS) 와 완속 충전소(Slow-charging Station, SS) 의 위치를 확인하였다.
#     - 급속: 파란색
#     - 완속: 초록색
# - 급속 충전소와 완속 충전소 주위 500m를 cover가능하다고 가정하였다.
# - 기존 충전소가 cover 가능한 point를 구분하였다.

# +
# 기존 완속/ 급속 충전소가 커버하는 위치 제거
df_01_geo = []
for i in range(len(df_01)):
    df_01_geo.append([df_01.loc[i,'충전소명'],Point(df_01.loc[i,'lon'],df_01.loc[i,'lat']).buffer(0.003)])
df_01[df_01['급속/완속']=='완속']
df_01_geo = pd.DataFrame(df_01_geo)
df_01_geo.columns = ["충전소명", "geometry"]
df_01_geo = pd.merge(df_01, df_01_geo, on = '충전소명')
df_01_geo['coordinates'] = df_01_geo['geometry'].apply(polygon_to_coordinates) 
df_01_geo = pd.DataFrame(df_01_geo)





center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
layer1 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_01_geo[df_01_geo['급속/완속']=='급속'][['coordinates']], # 시각화에 쓰일 데이터프레임
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[50, 50, 200,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

layer2 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_01_geo[df_01_geo['급속/완속']=='완속'][['coordinates']], # 시각화에 쓰일 데이터프레임
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[100, 200, 100,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

scatt1 = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='급속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=50,
    get_fill_color='[50, 50, 200]',
    pickable=True)

scatt2 = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='완속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=50,
    get_fill_color='[100, 200, 100]',
    pickable=True)


r = pdk.Deck(layers=[layer1,scatt1, layer2,scatt2], initial_view_state=view_state)
#             mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g" 
   
r.to_html()ㅠ

# +
#Fast-charging Station

#시간이 많이 걸립니다.
df_01_FS = df_01_geo[df_01_geo['급속/완속']=='급속']
FS_points = []
for i in tqdm(range(len(df_01_FS))):
    try:
        FS_points.append(point_cent.buffer(0.00000001).within(df_01_FS.loc[i,'geometry']))
    except: 
        pass
df_result['FS_station'] = 0
for i in range(len(FS_points)):
    df_result['FS_station'][FS_points[i]] = 1

#Slow-charging Station
df_01_SS = df_01_geo[df_01_geo['급속/완속']=='완속']    
SS_points = [] 
for i in tqdm(range(len(df_01_geo))):
    try:
        SS_points.append(point_cent.buffer(0.00000001).within(df_01_SS.loc[i,'geometry']))
    except:
        pass

df_result['SS_station'] = 0
for i in range(len(SS_points)):
    df_result['SS_station'][SS_points[i]] = 1

df_result.head()
