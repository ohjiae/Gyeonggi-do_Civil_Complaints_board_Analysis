#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chap02_step02_Dict_department
1. 부서(Key)와 관련 키워드(Value)를 가진 딕셔너리 만들기
2. 키워드(Value)를 통해 해당 부서(Key)를 찾는 함수 만들기
"""
import pandas as pd # csv file
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#부서별 키워드 정의
top9_dept = {
    '도로정책과' :['계획','다리','일산대교','착공','변경','이전','교통','통행료','한강','대교','도로'],
    '철도정책과' :['교통','북부','연장','착공','유치','신분당선','신설','면제','확정','지하철','호선','철도','노선', '통일로', '트램'],
    '버스정책과' :['교통','노선','변경','이전','교통카드','교통비','청소년','버스','대중교통','요금'],
    '철도건설과' :['교통','변경','이전','착공','원안','위치','호선','지하철','철도','노선', '트램'],
    '신도시추진단' :['시설','유치','이전','주택','지역','원안','위치','신도시', '하수'],
    '총무과' :['이재명','도지사','도지사님','지사님'],
    '감염병관리과' : ['코로나', '검사'],
    '신도시기획과' : ['병합', '부지', '폐기물', '시설', '위치'],
    '질병정책과' : ['코로나', '검사', '백신', '감염']
}

#부서별 키워드를 각 부서별로 하나의 리스트로 통합
top9 = []
for i in top9_dept.values():
    tmp = []
    j_tot = ""
    for j in i:
        tmp.append(i)
        j = j + " "
        j_tot += j
    top9.append(j_tot)

    
#본 파일(17000개)의 tfidf vectorizing    
tit_vectorizer = tfidf_vectorizer.fit_transform(tit_result)
tit_td = tit_vectorizer.toarray()

#부서별 키워드의 tfidf vectorizing
sm = tfidf_vectorizer.transform(top9)
sm = sm.toarray()

#부서별 키워드 vs 본 파일 title 간의 유사도 측정
test_sim = linear_kernel(tit_td, sm)

print(test_sim)

df = pd.DataFrame(test_sim, index = top9_dept.keys())

for i in range(17325) : 
    t = df.sort_values([i], ascending = False)

    if t.iloc[0,i] == 0 : 
        print(i+1, '번째 - 해당 부서 없음')
        
    else : 
        top = t.head(2)
        dept = top.index
            
        print(i+1, '번째 -', dict(zip(list(dept), [t.iloc[0,i], t.iloc[1,i]])))

