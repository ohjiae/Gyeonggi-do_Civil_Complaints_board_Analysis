# -*- coding: utf-8 -*-
"""
chap02_step00_Cosine_Similarity를 구하기 위한 여정 ~
"""
########### 주의 ##########
# <base>에 Jpype를 설치한 경우,
# <tensorflow> 환경에서는 konlpy 패키지가 작동하지 않을 수 있으므로
# 반드시 <tensorflow>에 먼저 Jpype를 설치한 환경에서만 작동하시길 바랍니다.

# 0. package load
# text 처리
import pandas as pd # csv file
import numpy as np
import string       # texts 전처리
from konlpy.tag import Okt

# 단어 빈도수 확인, 코사인 유사도
from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저
from sklearn.metrics.pairwise import cosine_similarity # 코사인 유사도
from sklearn.metrics.pairwise import linear_kernel


# 1. csv file load
#path = 'K:/ITWILL/Final_project/'
path = 'E:/ITWILL/Final_project/'
#minwon_data = pd.read_csv(path + 'crawlingdata17326.csv')
minwon_data = pd.read_csv(path + 'sep_crawling_data_17326.csv', encoding='CP949')
minwon_data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17326 entries, 0 to 17325
Data columns (total 4 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   Unnamed: 0  17326 non-null  int64 
 1   title       17326 non-null  object
 2   answer      17326 non-null  object
 3   sep         17326 non-null  int64 
'''

titles = minwon_data['title']
replies = minwon_data['answer']

#print(titles[:10])


# 2. 텍스트 전처리
# 1) 전처리 -> [text_sample.txt] 참고
# wc = what colomn( 어느 컬럼에서 전처리 돌릴 것인지)
def text_prepro(wc):
    # Lower case : 소문자
    wc = [x.lower() for x in wc]
    # Remove punctuation : 문장부호 제거
    wc = [''.join(c for c in x if c not in string.punctuation) for x in wc]
    # Remove numbers : 숫자 제거
    #titles = [''.join(c for c in x if c not in string.digits) for x in titles]
    # Trim extra whitespace : 공백 제거
    wc = [' '.join(x.split()) for x in wc]
    
    return wc


# 2) 함수 호출
# (1) titles 전처리
wc = titles
titles = text_prepro(wc)
print(titles[:10])


# 3. 불용어 제거 - Okt 함수 이용
# 1) 불용어 사전 - https://www.ranks.nl/stopwords/korean
korean_stopwords = path + "korean_stopwords.txt"

with open(korean_stopwords, encoding='utf8') as f : 
    stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]

#print(stopwords[:10])

# 2) 불용어 제거
okt = Okt()

tit_result = []  # 전처리 완료된 titles

# (1) titles 불용어 제거
for sentence in titles:  
  tmp = okt.morphs(sentence)
  #print('tmp :', tmp)
  tit_tokenized = []
  
  token_tot = ""    
  for token in tmp:      
      if not token in stopwords:
          tit_tokenized.append(token)
          #print('tit_tokenized :', tit_tokenized)
          token = token + " "
          token_tot += token
          #print('token_tot : ', token_tot)

  tit_result.append(token_tot)


len(tit_result) # 17326
print(tit_result[0])  # '경기도 지역화폐 사용 처가 너무 제한 적 입니다' 
print(tit_result[-1]) # '청소년 교통비'


# text vectorizing(tf-idf)
# 객체 생성
tfidf_vectorizer = TfidfVectorizer()
'''
# 문장 벡터화 진행 - 기존 코드
tfidf_matrix = tfidf_vectorizer.fit_transform(tit_result)  # 수정 
tfidf_matrix.shape # (17326, 3560)
# 각 단어
tit_word = tfidf_vectorizer.get_feature_names()
# get_feature_names()함수 설명 : tit_word에서 사용된 feature 단어 목록을 볼 수 있다.

# 각 단어 벡터값
tit_idf = tfidf_vectorizer.idf_
print(tit_idf)
len(tit_idf) # 3560

print(dict(zip(tit_word, tit_idf)))
'''

#############################################################################
### step01. 빈도수 상위 80개의 키워드 추출 (노가다 준비)
#############################################################################
# text vectorizing(tf-idf)
# 객체 생성
tfidf_vectorizer = TfidfVectorizer()

tfidf_often = TfidfVectorizer(max_features=80)
# max_features= : 사용할 최대 단어수를 지정
tfidf_often.fit(tit_result)                  #fit : 단어학습
top_tokens = tfidf_often.get_feature_names() # 단어목록 읽어오기
print(top_tokens)
# 나 혼자 그냥.. 시도...2줄()
tfidf_voca = tfidf_often.vocabulary_         # 학습한 단어사전 출력/ 여기위에 명령어를 붙여서 사용하는 구조
sorted(tfidf_often.vocabulary_.items())      # 순위 추출

# 각 단어
# tit_word = tfidf_vectorizer.get_feature_names()  
# print(tit_word)


##############################################################################
### step02~3 코사인 유사도
##############################################################################

"""
chap02_step02_Dict_department & Cosine_Similarity
1. 부서(Key)와 관련 키워드(Value)를 가진 딕셔너리 만들기
2. 키워드(Value)를 통해 해당 부서(Key)를 찾는 함수 만들기
"""

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


'''
함수 설명
.fit() : 단어 문서 행렬의 형태를 정하는 메소드로, 각 열에 들어갈 단어를 결정한다. 이 정보는 cv의 내부에 저장된다.
.transform() : .fit()으로 정한 단어에 데이터를 끼워 맞춰준다.
fit_transform은 fit과 transform  두 개의 메소드를 합친 것
'''

# 민원 파일(17000개)의 tfidf vectorizing
tit_vectorizer = tfidf_vectorizer.fit_transform(tit_result)
tit_td = tit_vectorizer.toarray()

# 부서별 키워드의 tfidf vectorizing
sm = tfidf_vectorizer.transform(top9)
sm = sm.toarray()

#부서별 키워드 vs 민원 파일 title 간의 유사도 측정
test_sim = linear_kernel(sm,tit_td) # 나 여기 linear_kernel 함수 아직 잘 모르겟엉
# 열 : 민원자료
# 행 : 담당부서
help(linear_kernel)
print(test_sim)

df = pd.DataFrame(test_sim, index = top9_dept.keys())

for i in range(len(tit_td)) : 
    t = df.sort_values([i], ascending = False)

    if t.iloc[0,i] == 0 : 
        print(i+1, '번째 - 해당 부서 없음')
        
    else : 
        top = t.head(2)
        dept = top.index
            
        print(i+1, '번째 -', dict(zip(list(dept), [t.iloc[0,i], t.iloc[1,i]])))
