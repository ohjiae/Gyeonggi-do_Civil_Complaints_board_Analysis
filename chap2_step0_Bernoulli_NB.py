# -*- coding: utf-8 -*-

"""
step00_naive_baise.py

민원게시글 나이브 베이즈로 0(중복아닌민원), 1(중복민원) 분류
"""
############## 베르누이 나이브 베이즈 ########################3
# 0. package load
# text 처리
import pandas as pd # csv file
import string       # texts 전처리
from konlpy.tag import Okt
import numpy as np

from sklearn.model_selection import train_test_split # dataset split
from sklearn.naive_bayes import BernoulliNB # model - 문서분류 
from sklearn.feature_extraction.text import TfidfVectorizer # 단어생성기 
from sklearn.metrics import confusion_matrix, classification_report # 평가 


# 파일 읽어오기 
# 1. csv file load
path = '/Users/aegohc/ITWILL/final_project/'
minwon_data = pd.read_csv(path + "sep_crawling_data_17326.csv", encoding='CP949')
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

titles = minwon_data['title']
# contents = minwon_data['contents']
replies = minwon_data['answer']
sep = minwon_data['sep']




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
 




# 3. 불용어 제거 - Okt 함수 이용
## 1) 불용어 사전 - https://www.ranks.nl/stopwords/korean
korean_stopwords = path + "korean_stopwords.txt"

with open(korean_stopwords, encoding='utf8') as f : 
    stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]
    
print(stopwords[:10])

## 2) 불용어 제거
okt = Okt()
tit_result = []  # 전처리 완료된 titles

### (1) titles 불용어 제거 - [수정]
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
print(tit_result[:10])




# 4. 모델 생성 및 학습
## 1) 데이터 타입 정리 후 X, Y 설정
ar_X = np.array(tit_result)
ar_Y = np.array(sep)
labels = ['일반민원','중복민원']
type(ar_X)

## 2) 변수 설정 및 데이터 벡터화
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(tit_result)
x_data = tfidf_vectorizer.transform(tit_result) #단어 카운트 가중치
y_data = ar_Y
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

## 3) 모델 생성 및 학습
model = BernoulliNB(alpha=1.0)
''' 매개변수 설명 : 이해 후 지우시면 됩니다
  alpha : float, optional (default=1.0)
  라플라스 스무딩  (과적합 방지)
  각 단어에 대한 확률의 분모, 분자에 전부 숫자를 더해서 분자가 0이 되는 것을 방지
'''
model.fit(x_train, y_train)

# 5. 모델 검증
# 1) score, Confusion matrix - Precision, Recall, F1-score, and Accuracy
print('Accuracy on training set:', model.score(x_train, y_train))
# Accuracy on training set: 0.9684201846965699
print('Accuracy on testing set:', model.score(x_test, y_test))
# Accuracy on testing set: 0.9630627164293959
y_predict_NB = model.predict(x_test)
confusion_matrix(y_test, y_predict_NB)
print(classification_report(y_test,y_predict_NB))
'''
              precision    recall  f1-score   support

           0       0.62      0.82      0.71       286
           1       0.99      0.97      0.98      4912

    accuracy                           0.96      5198
   macro avg       0.81      0.90      0.85      5198
weighted avg       0.97      0.96      0.97      5198
'''


## 2) 실제 데이터로 분류값 예
new_Xtest = np.array([
    '경기도 사람으로서 정말 행복합니다 화단 좀 가꾸어 주세요'
])
wc = x_test
new_Xtest = text_prepro(new_Xtest) #텍스트 정제
new_Xtest = tfidf_vectorizer.transform(new_Xtest) #단어 카운트 가중치

newY_predict = model.predict(new_Xtest)
label = labels[newY_predict[0]]
newY_predict = model.predict_proba(new_Xtest)
confidence = newY_predict[0][newY_predict[0].argmax()]

print(label, confidence)
# 중복민원 0.9573209328153289 
