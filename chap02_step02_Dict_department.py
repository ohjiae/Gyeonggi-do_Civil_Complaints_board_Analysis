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
test_sim = linear_kernel(sm, tit_td)

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

        
        
# 시각화 및 결과 저장

class_pred1 = []
class_pred2 = []

# aa = df.sort_values([7])

# bb = aa.head(2)
# dept = bb.index
# dept[0]
# dept[1]

for i in tqdm(range(17326)) : 
    t = df.sort_values([i], ascending = False)

    if t.iloc[0,i] == 0 : 
        print(i+1, '번째 - 해당 부서 없음')
        class_pred1.append('기타')
        class_pred2.append('기타')
    elif t.iloc[0,i] != 0 and t.iloc[1,i] == 0:
        top = t.head(1)
        dept = top.index
        class_pred1.append(dept[0])
        class_pred2.append('기타')
        print(i+1, '번째 -', dict(zip(list(dept), [t.iloc[0,i], t.iloc[1,i]])))
    else : 
        top = t.head(2)
        dept = top.index
        class_pred1.append(dept[0])
        class_pred2.append(dept[1])
        print(i+1, '번째 -', dict(zip(list(dept), [t.iloc[0,i], t.iloc[1,i]])))
        
print(list())



        
#modeling evaluation


# load 정답 data
rand_test = pd.read_csv('rand_test_deptno.csv', encoding=('CP949'))

rand_test.head()


# sim 유사도에 따른 예측 df화
cc = {'pred1' : class_pred1, 'pred2' : class_pred2}

class_pred = pd.DataFrame(cc)

# 부서명 코드화
class_pred.loc[class_pred.pred1 == '도로정책과', 'p1deptno'] = 1
class_pred.loc[class_pred.pred1 == '철도정책과', 'p1deptno'] = 2
class_pred.loc[class_pred.pred1 == '신도시추진단', 'p1deptno'] = 5
class_pred.loc[class_pred.pred1 == '버스정책과', 'p1deptno'] = 26
class_pred.loc[class_pred.pred1 == '철도건설과', 'p1deptno'] = 27
class_pred.loc[class_pred.pred1 == '총무과', 'p1deptno'] = 12
class_pred.loc[class_pred.pred1 == '감염병관리과', 'p1deptno'] = 30
class_pred.loc[class_pred.pred1 == '신도시기획과', 'p1deptno'] = 7
class_pred.loc[class_pred.pred1 == '질병정책과', 'p1deptno'] = 13
class_pred.loc[class_pred.pred1 == '기타', 'p1deptno'] = 0

class_pred.loc[class_pred.pred2 == '도로정책과', 'p2deptno'] = 1
class_pred.loc[class_pred.pred2 == '철도정책과', 'p2deptno'] = 2
class_pred.loc[class_pred.pred2 == '신도시추진단', 'p2deptno'] = 5
class_pred.loc[class_pred.pred2 == '버스정책과', 'p2deptno'] = 26
class_pred.loc[class_pred.pred2 == '철도건설과', 'p2deptno'] = 27
class_pred.loc[class_pred.pred2 == '총무과', 'p2deptno'] = 12
class_pred.loc[class_pred.pred2 == '감염병관리과', 'p2deptno'] = 30
class_pred.loc[class_pred.pred2 == '신도시기획과', 'p2deptno'] = 7
class_pred.loc[class_pred.pred2 == '질병정책과', 'p2deptno'] = 13
class_pred.loc[class_pred.pred2 == '기타', 'p2deptno'] = 0

class_pred.loc[:, 'p1deptno']
class_pred.loc[:, 'p2deptno']

rand_idx = rand_test.iloc[:, 0]

# rand_test의 랜덤 index의 예상 부서 코드를 추출
y_pred = class_pred.iloc[rand_idx]

# rand_test의 부서 코드 중 top9에 해당되지 않는 물건을 0처리
idx = []
for i, c in enumerate(rand_test.deptno) :
    if c not in (1, 2, 5, 26, 27, 12, 30, 7, 13) :
        idx.append(i)
    
rand_test.deptno[idx] = 0

y_test = rand_test.deptno

y_pred.iloc[:, 2]

accuracy_score(y_test, y_pred.iloc[:, 2]) #0.47
accuracy_score(y_test, y_pred.iloc[:, 3]) #0.15

confusion_matrix(y_test, y_pred.iloc[:, 2])
'''
array([[ 6,  1,  0, 12,  1,  4,  2,  0,  0],
       [ 0,  6,  0,  0,  0,  0,  0,  0,  0],
       [ 1,  1, 13,  0,  0,  0,  0,  0,  0],
       [ 2,  0,  0,  2,  0,  0,  0,  0,  0],
       [ 8, 10,  0,  9, 16,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  1],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  1,  0,  0,  0,  4,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=int64)
'''
confusion_matrix(y_test, y_pred.iloc[:, 3])

'''
array([[14,  2,  9,  0,  0,  0,  0,  0,  1],
       [ 4,  0,  2,  0,  0,  0,  0,  0,  0],
       [ 7,  1,  1,  5,  0,  0,  0,  1,  0],
       [ 2,  0,  0,  0,  0,  0,  0,  2,  0],
       [17,  0,  0, 26,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  1,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  5,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=int64)
'''

