# -*- coding: utf-8 -*-
######################## chap1_step2부분이 앞에 포함되어 연결됩니다. #####################################

# 0. package load
# text 처리
import pandas as pd # csv file
import string       # texts 전처리
from konlpy.tag import Okt

# 단어 빈도수 확인, 코사인 유사도
from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저
from sklearn.metrics.pairwise import cosine_similarity # 코사인 유사도


# 1. csv file load
path = '/Users/aegohc/ITWILL/final_project/'
data = pd.read_csv(path + "sep_crawling_data_17326.csv", encoding='CP949')
data.info()

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

titles = data['title']
replies = data['answer']
sep = data['sep']
#print(titles[:10])



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


# 2) 함수 호출
# (1) titles 전처리
wc = titles
titles = text_prepro(wc)
print(titles[:10])
'''
# (2) replies 전처리
wc = replies
replies = text_prepro(wc)
print(replies[:10])
'''


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
print(tit_result[0])  # '경기도 지역화폐 사용 처가 너무 제한 적 입니다' 
print(tit_result[-1]) # '청소년 교통비'

'''
# (2) replies 불용어 제거
for sentence in replies:
  tmp = []
  tmp = okt.morphs(sentence)
  
  rpl_tokenized = []
  for token in tmp:
    if not token in stopwords:
      tit_tokenized.append(token)


  rpl_result.append(rpl_tokenized)

print(rpl_result[:10])
'''
'''
# 4. csv file save
tit_result = pd.DataFrame(tit_result)
tit_result.to_csv('titles_preprocessing.csv', index = None, encoding = 'utf-8')
print(tit_result[:5])
'''
'''
rpl_result = pd.DataFrame(rpl_result)
rpl_result.to_csv('titles_preprocessing.csv', index = None, encoding = 'utf-8')
'''
#tit_result.shape # (17326, 40)


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

#%%
######################################################### 여기서부터 Bernoulli NB 시작 #############################################################

"""
chap02
step02_Naive_Baise.py

MODEL CASE32337
1. SVM           - 지은님
2. Naive Baise   - 지애님
3. Decision tree - 다현님

# 베르누이 나이브 베이즈로 0(일반민원), 1(중복민원) 분류
"""

# 환경설정
import numpy as np
from sklearn.model_selection import train_test_split     # dataset split
from sklearn.naive_bayes import BernoulliNB # 베르누이 나이브 베이즈 모델
from sklearn.naive_bayes import MultinomialNB # 다항 분포 나이브 베이즈 모델
from sklearn.metrics import confusion_matrix, classification_report  # 평가

# 1. dataset load
path = '/Users/aegohc/ITWILL/final_project/'
data = pd.read_csv(path + "sep_crawling_data_17326.csv", encoding='CP949')
data.info()
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

# 2. train/test split 
labels = ['일반민원','중복민원']
x_train, x_test, y_train, y_test = train_test_split(tfidf_matrix, sep, test_size=0.3, random_state=777, stratify=sep)

# 3. 베르누이 나이브 베이즈 모델
# 1) 모델 훈련
''' 매개변수 설명 : 
  alpha : float, optional (default=1.0)
  라플라스 스무딩  (과적합 방지)
  각 단어에 대한 확률의 분모, 분자에 전부 숫자를 더해서 분자가 0이 되는 것을 방지
'''
model = BernoulliNB(alpha=1.0)
model.fit(x_train, y_train)

# 2) 모델 평가(Accuracy, Confusion matrix - Precision, Recall, F1-score)
print('Accuracy on training set:', model.score(x_train, y_train))
# Accuracy on training set: 0.9684201846965699
print('Accuracy on testing set:', model.score(x_test, y_test))
# Accuracy on testing set: 0.9630627164293959
y_predict_NB = model.predict(x_test)
con_mat = confusion_matrix(y_test, y_predict_NB)
print(con_mat)
'''
[[ 235   51] 
 [ 141 4771]]
'''
print(classification_report(y_test,y_predict_NB))
'''
              precision    recall  f1-score   support

           0       0.62      0.82      0.71       286
           1       0.99      0.97      0.98      4912

    accuracy                           0.96      5198
   macro avg       0.81      0.90      0.85      5198
weighted avg       0.97      0.96      0.97      5198
'''


# 3) 실제 데이터로 분류값 예
new_Xtest = np.array([
    '경기도 사람으로서 정말 행복합니다 화단 좀 가꾸어 주세요'
])
wc = new_Xtest
new_Xtest = text_prepro(new_Xtest) #텍스트 정제
new_Xtest = tfidf_vectorizer.transform(new_Xtest) #단어 카운트 가중치

newY_predict = model.predict(new_Xtest)
label = labels[newY_predict[0]]
newY_predict = model.predict_proba(new_Xtest)
confidence = newY_predict[0][newY_predict[0].argmax()]

print(label, confidence)
# 중복민원 0.9573209328153289  # 심지어 틀림 ㅠㅠ


#%%
# 4. 다항 분포 나이브 베이즈 모델 (Multinomial Naive Baise)
# 데이터 로드와 x,y 데이터 스플릿 등은 베르누이 모델에서 이미 했으므로 생략함.

mnb_model = MultinomialNB(alpha = 1.0)
mnb_model.fit(x_train, y_train)

# 2) 모델 평가(Accuracy, Confusion matrix - Precision, Recall, F1-score)
print('Accuracy on training set:', mnb_model.score(x_train, y_train))
# Accuracy on training set: 0.9801286279683378
print('Accuracy on testing set:', mnb_model.score(x_test, y_test))
# Accuracy on testing set: 0.974028472489419
y_predict_mnb = mnb_model.predict(x_test)
con_mat = confusion_matrix(y_test, y_predict_mnb)
print(con_mat)
'''
[[ 193   93]
 [  42 4870]]
'''
print(classification_report(y_test,y_predict_mnb))
'''
              precision    recall  f1-score   support

           0       0.82      0.67      0.74       286
           1       0.98      0.99      0.99      4912

    accuracy                           0.97      5198
   macro avg       0.90      0.83      0.86      5198
weighted avg       0.97      0.97      0.97      5198
'''


# 3) 실제 데이터로 분류값 예
mnb_Xtest = np.array([
    '제발 좀 맞춰 줄세요. 아무리 0값이 없다지만 어디 사는지도 안 들어 갔는데 못 맞추면 슬픕니다'
])
wc = mnb_Xtest
mnb_Xtest = text_prepro(mnb_Xtest) #텍스트 정제
mnb_Xtest = tfidf_vectorizer.transform(mnb_Xtest) #단어 카운트 가중치

mnb_Y_predict = mnb_model.predict(mnb_Xtest)
mnb_label = labels[mnb_Y_predict[0]]
mnb_Y_predict = mnb_model.predict_proba(mnb_Xtest)
mnb_confidence = mnb_Y_predict[0][mnb_Y_predict[0].argmax()]

print(mnb_label, mnb_confidence)
# 중복민원 0.987109181471525  # 또 틀림 ㅠㅠ


# 베르누이 나이브 베이즈 모델 / 다항 분포 나이브 베이즈 모델이었습니다.