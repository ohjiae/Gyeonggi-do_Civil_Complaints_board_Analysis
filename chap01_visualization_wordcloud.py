"""
visualization_wordcloud.py
"""
# 0. package load
# text 처리
import pandas as pd # csv file
import numpy as np
import string       # texts 전처리

import os
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud

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
#rpl_result = []  # 전처리 완료된 replies

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

len(tit_result) # 17326
#print(tit_result[0])  # '경기도 지역화폐 사용 처가 너무 제한 적 입니다' 


'''
# (1-2) 결과값 csv로 저장 및 불러오기
df = pd.DataFrame(tit_result) #여기만하고 넘어가기... 오류생겨
df.to_csv('tit_result.csv', index=False)
tit_result = pd.read_csv('tit_result.csv', encoding=('utf-8'))
tit_result = np.array(tit_result).flatten().tolist()
'''


# Okt 형태소 분석 객체 생성
ok_tit = Okt()

#형태소 분류하고 확인하기
sentences_tag = []
for sentence in tit_result:
  morph = ok_tit.pos(sentence)
  sentences_tag.append(morph)
  # print(morph) 
  #print('-' * 30)
'''
[('위례간담회', 'Noun'), ('에', 'Josa'), ('이재명', 'Noun'), ('도지사', 'Noun'), ('님', 'Suffix'), ('꼭', 'Noun'), ('참석', 'Noun'), ('부탁드립니다', 'Adjective')]
'''

# 필요한 품사만 추출(명사를 bucket list에 담기)
bucket_list = []
for my_sentence in sentences_tag:
  for word, tag in my_sentence:
    if tag in ['Noun']:
        bucket_list.append(word)
#print(bucket_list)


# 단어 빈도수 구하기 
# 각 원소의 출현 횟수를 계산하는 counter 모듈 활용
counts = Counter(bucket_list)
print(counts)

# 명사 빈도 순서대로 상위 100개 출력
top100_word = counts.most_common(100)
print(top100_word)

''' 명사만 추출한 결과와 거의 똑같았음
# 명사와 형용사를 모두 추출하고 상위 50개를 출력
bucket_list_2 = []
for my_sentence in sentences_tag:
  for word, tag in my_sentence:
    if tag in ['Noun','Adjective']:
      bucket_list_2.append(word)
counts = Counter(bucket_list)
print(counts.most_common(50))
'''

# 단어 구름 시각화
# 경기도 지도 : word cloud
from PIL import Image
# 경기도 지도 이미지 불러오기
mask_image = np.array(Image.open('E:/ITWILL/Final_project/kungkii_map2.jpg'))

wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf',
          width=800, height=977,
          max_words=100,max_font_size=450,
          mask=mask_image,
          background_color='white',
          colormap='Accent')

# top100_word : list형 -> dict형으로!(딕셔너리 자료형)
wc_result = wc.generate_from_frequencies(dict(top100_word))

import matplotlib.pyplot as plt
plt.imshow(wc_result)
plt.axis('off') # 축 눈금 감추기
plt.show()

#
#wc.to_file('wordcloud.png')
