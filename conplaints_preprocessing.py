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
import numpy as np
import string       # texts 전처리
from konlpy.tag import Okt


# 1. csv file load
path = 'D:/서다현/빅데이터/Final_Project/ITWILL-Final_project-main/'
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
replies = minwon_data['answer']
sep = minwon_data['sep']

print(titles)


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
print(titles)

# replies 전처리
replies = text_prepro(replies)
print(replies)


# 3. 불용어 제거 - Okt 함수 이용
# 1) 불용어 사전 - https://www.ranks.nl/stopwords/korean
korean_stopwords = path + "korean_stopwords.txt"

with open(korean_stopwords, encoding='utf8') as f : 
    stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]
    
print(stopwords[:10])

# 2) 불용어 제거
okt = Okt()
'''
    여기서 (1)과 (2)를 한번에 돌리면 reply 불용어만 제거되므로
    (1)titles 돌려서 제거 후 (2)replies 돌려야함
'''
tit_result = []  # 전처리 완료된 titles
rpl_result = []  # 전처리 완료된 replies


### (1) titles 불용어 제거
for sentence in titles:  
  tmp = okt.morphs(sentence)
  tit_tokenized = []
  
  token_tot = ""    
  for token in tmp:      
      if not token in stopwords:
          tit_tokenized.append(token)
          token = token + " "
          token_tot += token

  tit_result.append(token_tot)

print(tit_result)


''' 여기 위쪽 먼저 돌린 후 아래 돌리세요 '''


### (2) replies 불용어 제거
for sentence in replies:  
  tmp = okt.morphs(sentence)
  rpl_tokenized = []
  
  token_tot = ""    
  for token in tmp:      
      if not token in stopwords:
          rpl_tokenized.append(token)
          token = token + " "
          token_tot += token

  rpl_result.append(token_tot)

print(rpl_result)

'''
# 4. csv file save - 생략 가능
# titles 저장
titles = pd.DataFrame(tit_result)
titles.to_csv('titles.csv', index = None, encoding = 'CP949')

# replies 저장
replies = pd.DataFrame(rpl_result)
replies.to_csv('replies.csv', index = None, encoding = 'CP949')
'''
