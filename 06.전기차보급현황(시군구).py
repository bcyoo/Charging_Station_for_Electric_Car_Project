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
# 한글폰트 사용
import platform
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    f_path = '/Library/Fonts/Arial Unicode.ttf'
elif platform.system() == 'Windows':
    f_path = 'c:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=f_path).get_name()
rc('font', family=font_name)

print('Hangul font is set!')

# +
# 필요패키지 import
import numpy as np
import pandas as pd
from datetime import datetime
import csv # csv 파일 저장
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import * # 데이터 시각화 
import re # 정규표현식
# 지도 표현
import folium

# %matplotlib inline 
# -

import networkx as nx 
from geopy.geocoders import Nominatim
import osmnx as ox 
import shapely 
import pandas as pd
from shapely import geometry
from descartes.patch import PolygonPatch
from shapely.geometry import LineString

# +
## 시군구 자동차 등록 현황 csv 파일 불러오기
df = pd.read_csv('DATA/_전기차보급현황/06.전기차보급현황(시군구).csv', engine='python')

## 시군구 col 만들고 시 col + 군구 col
df["시군구"] = df["시도"].map(str) + " " + df["지역구분"]

## 컬럼명 재설정
df = df[['시도', '지역구분', '차종', '전체', '차종', '전체', '차종', '전체', '총합', '시군구']]
df

# -

df.loc[df.지역구분 == '서울특별시', ('시군구')] = ('서울특별시')
df.loc[df.지역구분 == '부산광역시', ('시군구')] = ('부산광역시')
df.loc[df.지역구분 == '대구광역시', ('시군구')] = ('대구광역시')
df.loc[df.지역구분 == '인천광역시', ('시군구')] = ('인천광역시')
df.loc[df.지역구분 == '광주광역시', ('시군구')] = ('광주광역시')
df.loc[df.지역구분 == '대전광역시', ('시군구')] = ('대전광역시')
df.loc[df.지역구분 == '울산광역시', ('시군구')] = ('울산광역시')
df.loc[df.지역구분 == '세종특별자치시', ('시군구')] = ('세종특별시')
df.loc[df.지역구분 == '제주특별자치도', ('시군구')] = ('제주특별자치도')

df.info()

## 지오코딩에 필요한 리스트 시군구 생성
address = df['시군구'].tolist()
address

# +
import googlemaps
import pandas as pd

my_key = "AIzaSyCUKB2z6sahUkbf-MmWmpOf7ZM56Mte118"
maps = googlemaps.Client(key=my_key)  # my key값 입력
lat = []  #위도
lng = []  #경도

# 위치를 찾을 장소나 주소를 넣어준다.
places = '세종특별시'

i=0
for place in places:   
    i = i + 1
    try:
        print("%d번 인덱스에서 %s의 위치를 찾고있습니다"%(i, place))
        geo_location = maps.geocode(place)[0].get('geometry')
        lat.append(geo_location['location']['lat'])
        lng.append(geo_location['location']['lng'])
        

    except:
        lat.append('')
        lng.append('')
        print("%d번 인덱스 위치를 찾는데 실패했습니다."%(i))


# 데이터프레임만들어 출력하기
df_1 = pd.DataFrame({'위도':lat, '경도':lng}, index=places)
print(df_3)
# -

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

## 경도 위도 DataFrame 재설정
df_1.info()
df_1

## index 재설정
df_1 = df_1.reset_index()

df_1 = df_1.rename(columns={'index' : '시군구'})

df_2 = pd.merge(df, df_1,how='outer',on='시군구')
df_2

(ggplot(df_2)  ## gg plot 으로 시각화
 + aes(x='경도', y='위도', color='총합')
 + geom_point() # 점포인트로 찍음
 + theme(text=element_text(family=font_name))
) 

df_2.dtypes

df_2 = df_2.astype({'총합':'str'})
df_2

df_2.loc[df_2.시군구 == '세종특별자치시', ('위도', '경도')] = (36.48, 127.29)

# +
df_2.shape
map = folium.Map(location=[df_2['위도'].mean(), df_2['경도'].mean()], zoom_start=13)

for n in df_2.index:
    park_name = df_2.loc[n, '총합'] + ' - ' + df_2.loc[n, '시군구']
    folium.Marker([df_2.loc[n, '위도'], df_2.loc[n, '경도']], popup=park_name).add_to(map)
map
# -
df_2.to_csv('./06.전기차보급현황(시군구).csv',index=False, encoding='cp949')
