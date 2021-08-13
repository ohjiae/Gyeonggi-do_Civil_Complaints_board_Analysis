######################################### 여기는 랜덤 변수 뽑고 파일 생성하는 과정입니다. 파일 다운 후 중간 코드부터 실행하세요! ##############################################
"""
chap02_visualization_test


[시각화 제안1] 성능 측정
 -> 대충 100개정도(random추출필요) 정답데이터 필요

1. 데이터를 100개를 랜덤추출
2. 100개에 부서 코드를 입력해 정답데이터 생성 (정답데이터 : rand_test)
3. 모델로 돌려서 나온 값..이거랑 정답데이터 비교

"""
# 0. package load
# text 처리
import pandas as pd # csv file
import numpy as np
import string       # texts 전처리
from konlpy.tag import Okt

# 단어 빈도수 확인, 코사인 유사도
from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저
from sklearn.metrics.pairwise import cosine_similarity # 코사인 유사도

# 1. csv file load
#path = 'K:/ITWILL/Final_project/'
path = 'E:/ITWILL/Final_project/'
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
sep = minwon_data['sep']


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
#print(titles[:10])


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

  #tit_result.append(tit_tokenized)
  tit_result.append(token_tot)


# 변수 생성
import random
np.random.seed(123)
rand_index = np.random.randint(len(minwon_data),size=100)
# print(rand_index)

rand_test = minwon_data.iloc[rand_index]
rand_test.head()

rand_test.to_csv('rand_test.csv', index=False, encoding='cp949')
################################################################## 파일 다운 후 여기부터 성능 test 코드 ##########################################################################

# deptno 부서코드 추가처리 된 상태이다.
rand_test = pd.read_csv(path + 'rand_test_deptno.csv', encoding='CP949')
rand_test.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   Unnamed: 0  100 non-null    int64 
 1   title       100 non-null    object
 2   answer      100 non-null    object
 3   sep         100 non-null    int64 
 4   deptno      100 non-null    int64 
'''
