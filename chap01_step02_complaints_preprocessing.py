# -*- coding: utf-8 -*-

"""
chap01_step02_complaints_preprocessing.py

민원 전처리
"""

# 0. package load
# text 처리
import pandas as pd # csv file
import numpy as np
import string       # texts 전처리
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저


# 1. csv file load
path = 'C:/Users/STU-16/Desktop/빅데이터/Final_Project/ITWILL-Final_project-main/' # 디렉토리 환경에 맞게 수정
minwon_data = pd.read_csv(path + 'sep_crawling_data_17326.csv', encoding = 'CP949')
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
sep = minwon_data['sep']

print(titles[:10])


# 2. sep, titles, replies 전처리
# 1) sep 전처리
sep = np.array(sep)

# 2) 전처리 함수 -> [text_sample.txt] 참고
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

# 2) 함수 호출
# titles 전처리
titles = text_prepro(titles)
print(titles[:10])


# 3. 불용어 제거 - Okt 함수 이용
# 1) 불용어 사전 - https://www.ranks.nl/stopwords/korean
korean_stopwords = path + "korean_stopwords.txt"

with open(korean_stopwords, encoding='utf8') as f : 
    stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]
    
print(stopwords[:10])

# 2) 불용어 제거
okt = Okt()

tit_result = []  # 전처리 완료된 titles


# titles 불용어 제거
for sentence in titles:  
  tmp1 = okt.morphs(sentence)
  tit_tokenized = []
  
  token_tot = ""    
  for token in tmp1:      
      if not token in stopwords:
          tit_tokenized.append(token)
          token = token + " "
          token_tot += token

  tit_result.append(token_tot)

print(tit_result[:10])

'''
# 4. csv file save - 생략 가능
# titles 저장
titles = pd.DataFrame(tit_result)
titles.to_csv('titles.csv', index = None, encoding = 'CP949')
'''

# 5. text vectorizing(tf-idf)
# 1) 객체 생성
tfidf_vectorizer = TfidfVectorizer()           

# 2) titles 문장 벡터화 진행
# 문장 벡터화 진행
tit_vectorizer = tfidf_vectorizer.fit_transform(tit_result)

# 각 단어 벡터화 진행
tit_word = tfidf_vectorizer.get_feature_names()  
# 각 단어 벡터값
tit_idf = tfidf_vectorizer.idf_

# 단어, IDF 값 매칭 리스트
tit_idf_list = dict(zip(tit_word, tit_idf)) 
# 단어와 부여된 정수값 확인
tit_index = tfidf_vectorizer.vocabulary_

print(tit_index[:10])
