# -*- coding: utf-8 -*-

"""
conplaints_preprocessing.py

민원 전처리
"""

# 0. package load
# text 처리
import pandas as pd # csv file
import string       # texts 전처리
from konlpy.tag import Okt


# 1. csv file load
#path = 'K:/ITWILL/Final_project/'
path = 'C:/Users/STU-16/Desktop/빅데이터/Final_Project/ITWILL-Final_project-main/'
minwon_data = pd.read_csv(path + 'crawlingdata17326.csv', header = None)
minwon_data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17327 entries, 0 to 17326
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0       17326 non-null  float64
 1   1       17327 non-null  object 
 2   2       17327 non-null  object 
'''

titles = minwon_data[1]
replies = minwon_data[2]

print(titles)


# 2. titles 전처리
# 1) titles 전처리 -> [text_sample.txt] 참고
def text_prepro(titles):
    # Lower case : 소문자
    titles = [x.lower() for x in titles]
    # Remove punctuation : 문장부호 제거
    titles = [''.join(c for c in x if c not in string.punctuation) for x in titles]
    # Remove numbers : 숫자 제거
    titles = [''.join(c for c in x if c not in string.digits) for x in titles]
    # Trim extra whitespace : 공백 제거
    titles = [' '.join(x.split()) for x in titles]
    return titles

# 2) 함수 호출
titles = text_prepro(titles)
print(titles)


# 3. 불용어 제거 - Okt 함수 이용
# 1) 불용어 사전 - https://www.ranks.nl/stopwords/korean
korean_stopwords = "C:/Users/STU-16/Desktop/빅데이터/Final_Project/ITWILL-Final_project-main/korean_stopwords.txt"

with open(korean_stopwords, encoding='utf8') as f : 
    stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]
    
print(stopwords[:10])

# 2) 불용어 제거
okt = Okt()

result = []
for sentence in titles:
  tmp = []
  tmp = okt.morphs(sentence)
  
  tit_tokenized = []
  for token in tmp:
    if not token in stopwords:
      tit_tokenized.append(token)


  result.append(tit_tokenized)

print(result)

# 4. csv file save
result = pd.DataFrame(result)
result.to_csv('titles.csv', index = None, encoding = 'utf-8')
