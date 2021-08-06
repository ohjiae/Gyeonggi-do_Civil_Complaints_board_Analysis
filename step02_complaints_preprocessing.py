# -*- coding: utf-8 -*-

"""
conplaints_preprocessing.py
민원 전처리
"""
########### 주의 ##########
# <base>에 Jpype를 설치한 경우, <tensorflow> 환경에서는 
# konlpy 패키지가 작동하지 않을 수 있으므로
# 반드시 <tensorflow>에 먼저 Jpype를 설치한 환경에서만 작동하시거나
# spyder를 anaconda3로 작동해주시길 바랍니다.


# 0. package load
# text 처리
import pandas as pd # csv file
import string       # texts 전처리
from konlpy.tag import Okt


# 단어 빈도수 확인, 코사인 유사도

from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저
from sklearn.metrics.pairwise import cosine_similarity # 코사인 유사도


# 파일 읽어오기 
# 1. csv file load
path = '/Users/aegohc/ITWILL/final_project/'
minwon_data = pd.read_csv(path + 'minwon_crawling4400.csv')
minwon_data.info()

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4400 entries, 0 to 4399
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   titles    4400 non-null   object 
 1   contents  0 non-null      float64
 2   replies   4400 non-null   object 
 3   sep       4399 non-null   float64
dtypes: float64(2), object(2)
memory usage: 137.6+ KB
'''

titles = minwon_data['titles']
contents = minwon_data['contents']
replies = minwon_data['replies']
sep = minwon_data['sep']


print(titles[:10])




# 2. 텍스트 전처리
## 1) 전처리 -> [text_sample.txt] 참고
## wc = what colomn (어느 컬럼에서 전처리 돌릴 것인지)
def text_prepro(wc):
    # Lower case : 소문자
    wc = [x.lower() for x in wc]
    # Remove punctuation : 문장부호 제거
    wc = [''.join(c for c in x if c not in string.punctuation) for x in wc]
    # Remove numbers : 숫자 제거 [생략] (추후 'n호선' 사용 여부)
    #wc = [''.join(c for c in x if c not in string.digits) for x in wc]
    # Trim extra whitespace : 공백 제거
    wc = [' '.join(x.split()) for x in wc]

    return wc

## 2) 함수 호출
### (1) titles 전처리
wc = titles
titles = text_prepro(wc)
print(titles[:10])

### (2) replies 전처리
wc = replies
replies = text_prepro(wc)
print(replies[:10])


# 3. 불용어 제거 - Okt 함수 이용
## 1) 불용어 사전 - https://www.ranks.nl/stopwords/korean
korean_stopwords = path + "korean_stopwords.txt"

with open(korean_stopwords, encoding='utf8') as f : 
    stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]
    
print(stopwords[:10])

## 2) 불용어 제거
okt = Okt()
'''
    여기서 (1)과 (2)를 한번에 돌리면 reply 불용어만 제거되므로
    (1)titles 돌려서 제거 후 (2)replies 돌려야함
'''
tit_result = []  # 전처리 완료된 titles
rpl_result = []  # 전처리 완료된 replies

'''
### (1) titles 불용어 제거
for sentence in titles:
  tmp = []
  tmp = okt.morphs(sentence)
  
  tit_tokenized = []
  for token in tmp:
    if not token in stopwords:
      tit_tokenized.append(token)

  tit_result.append(tit_tokenized)
  
 print(tit_result[:10]) 
'''
# (1) titles 불용어 제거 - [수정]
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

len(tit_result) # 17326
print(tit_result[0])  # '경기도 지역화폐 사용 처가 너무 제한 적 입니다' 
print(tit_result[-1]) # '청소년 교통비'



''' 여기 위쪽 먼저 돌린 후 아래 돌리세요 '''


### (2) replies 불용어 제거
for sentence in replies:
  tmp = []
  tmp = okt.morphs(sentence)
  
  rpl_tokenized = []
  for token in tmp:
    if not token in stopwords:
      rpl_tokenized.append(token)

  rpl_result.append(rpl_tokenized)

print(rpl_result[:10])


'''
# 5. 벡터라이징 (tfidf)
# 객체 생성 
tfidf_vectorizer = TfidfVectorizer()

### (1) titles 문장 벡터화 진행
for k in range(len(tit_result)) :
    if k < len(tit_result):
        tit_after_vec = tfidf_vectorizer.fit_transform(tit_result[k]) # titles 단어 학습
        tit_word = tfidf_vectorizer.get_feature_names() # 각 단어 토큰
        tit_idf = tfidf_vectorizer.idf_ # 각 단어의 idf 값
        len(tit_idf) #  1573

        tit_result.append(tit_after_vec)
        k += 1
    else:
        tit_after_vec = tfidf_vectorizer.fit_transform(tit_result[k]) # titles 단어 학습
        tit_word = tfidf_vectorizer.get_feature_names() # 각 단어 토큰
        tit_idf = tfidf_vectorizer.idf_ # 각 단어의 idf 값
        len(tit_idf) #  1573

        tit_result.append(tit_after_vec)
        break

print(tit_result[:10])
'''

# 5.text vectorizing(tf-idf)
# 객체 생성
tfidf_vectorizer = TfidfVectorizer()           

# 문장 벡터화 진행
tfidf_matrix = tfidf_vectorizer.fit_transform(tit_result)  # 수정 
tfidf_matrix.shape # (17326, 3560)
# 각 단어
tit_word = tfidf_vectorizer.get_feature_names()  

# 각 단어 벡터값
tit_idf = tfidf_vectorizer.idf_
print(tit_idf)
len(tit_idf) # 3560

print(dict(zip(tit_word, tit_idf)))







'''
    tfidf_vectorizer 객체 변수가 같으므로 
    (1)titles 돌려서 tit_index 변수 안에 데이터 들어간 상태에서
    (2)replies 돌릴 것.
'''
# list 한문장 기준 실행되어야 하는 내용
tit_vectorizer = tfidf_vectorizer.fit_transform(tit_result[1]) # titles 단어 학습
tit_word = tfidf_vectorizer.get_feature_names() # 각 단어 토큰
tit_idf = tfidf_vectorizer.idf_ # 각 단어의 idf 값
len(tit_idf) #  1573

# 단어, IDF 값 매칭 리스트
tit_idf_list = dict(zip(tit_word, tit_idf)) 
# 확인 필요시 출력
# print(tit_idf_list)

# 단어와 부여된 정수값 확인
tit_index = tfidf_vectorizer.vocabulary_
# 확인 필요시 출력
# print(tit_index)


''' 여기 위쪽 먼저 돌린 후 아래 돌리세요 '''


### (2) replies 문장 벡터화 진행
rpl_vectorizer = tfidf_vectorizer.fit_transform(rpl_result) # replies 단어 학습
rep_word = tfidf_vectorizer.get_feature_names() # 각 단어 토큰
rep_idf = tfidf_vectorizer.idf_ # 각 단어의 idf 값
len(rep_idf) # 4654 

# 단어, IDF 값 매칭 리스트
rpl_vec_list = dict(zip(rep_word, rep_idf))
# 확인 필요시 출력
# print(rpl_vec_list)

# 단어와 부여된 인덱스 확인
rpl_index = tfidf_vectorizer.vocabulary_
# 확인 필요시 출력
# print(rpl_index)

'''
# 6. 각 문장을 나머지 문장과 비교해 코사인 유사도 구하기
## 1) titles 리스트의 문장들 비교
for i in  range(len(tit_result)): # 0 ~ 99
    for j in range(len(tit_res)): # 1 ~ 99
        if i == j :
            break
        else :           
            cosine_similarity(tit_res[i], tit_res[j])

cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])


'''


