# -*- coding: utf-8 -*-
######################## chap1_step2부분이 앞에 포함되어 연결됩니다.[불용어 파일 저장했다가 불러올때, df -> list로 변환하는 과정 없어서요! #####################################




"""
conplaints_preprocessing.py
민원 전처리
"""
########### 주의 ##########
# <base>에 Jpype를 설치한 경우,
# <tensorflow> 환경에서는 konlpy 패키지가 작동하지 않을 수 있으므로
# 반드시 <tensorflow>에 먼저 Jpype를 설치한 환경에서만 작동하시길 바랍니다.

# 0. package load
# text 처리
import pandas as pd # csv file
import string       # texts 전처리
from konlpy.tag import Okt

# 단어 빈도수 확인, 코사인 유사도
from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저
from sklearn.metrics.pairwise import cosine_similarity # 코사인 유사도

# 1. csv file load
#path = 'K:/ITWILL/Final_project/'
path = 'E:/ITWILL/Final_project/'
minwon_data = pd.read_csv(path + 'crawlingdata17326.csv')
#minwon_data = pd.read_csv(path + 'sep_crawling_data_17326.csv')
#minwon_data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17326 entries, 0 to 17325
Data columns (total 3 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   Unnamed: 0  17326 non-null  int64 
 1   title       17326 non-null  object
 2   answer      17326 non-null  object
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
    
#print(stopwords[:10])

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

#################################################################### 여기서부터 SVM modeling 시작 #####################################################################
"""
chap02
step00_SVM_model.py

MODEL CASE32337
1. SVM           - 지은님
2. Naive Baise   - 지애님
3. Decision tree - 다현님

# SVM모델은 선형 & 비선형
Hyper parameger : kernel, C, gamma
"""
# 환경설정
import pandas as pd                                      # csv file
from sklearn.svm import SVC                              # svm model
from sklearn.model_selection import train_test_split     # dataset split
from sklearn.metrics import accuracy_score, confusion_matrix   # 평가

# 1. dataset load
#path = 'E:/ITWILL/Final_project/'
data = pd.read_csv(path +"sep_crawling_data_17326.csv", encoding='CP949')
#data.info()
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
sep = data['sep']
#data.head()
#tit_result = pd.read_csv(path + 'titles_preprocessing(numx).csv')

sep.shape         # (17326,) 
sep.value_counts()
'''
1    16373 - 중복민원
0      953 - 비 중복민원
'''

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, sep, test_size=0.3, random_state=123)

X_train.shape # (12128, 3560)
X_test.shape  # (5198, 3560) 



# 3. 비선형 SVM 모델
# help(SVC)
obj = SVC(C=1.0, kernel='rbf', gamma='scale') # 가장 중요한 3개의 parameter
'''
[비선형 SVM] -> default 기본 parameter
C=1.0 : cost(오분류)조절 : 결정경계 [위치]조정(값이 클수록 정화도 높음, 값이 작을 수록 정확도가 낮음)

kernel ='rbf' : 커널트릭 함수 
 -> kernel : {'linear'(선형), 'poly', 'rbf'(비선형), 'sigmoid', 'precomputed'}, default='rbf'

gamma ='scale' : 결정경계 [모양]을 조정
 -> {'scale', 'auto'} or float
 -> gamma ='scale' : 1 / (n_features * X.var())
 -> gamma ='auto' : 1 / n_features
 -> gamma = 0.1
 n_features : x변수 개수
'''

model = obj.fit(X=X_train, y=y_train)

# model 평가
y_pred = model.predict(X = X_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy = ',acc)
# accuracy =  0.9782608695652174

con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)
'''
0 [[ 206   93]
1 [  20 4879]
'''
206 / (206  + 93) # 0.6889632107023411 -> 0 예측력
4879/(20 + 4879)  # 0.9959175341906511


# 4. 선형 SVM : 선형분류 가능한 데이터 (noise 없는 데이터)
obj2 = SVC(C=1.0, kernel='linear', gamma='scale') # 가장 중요한 3개의 parameter
model2 = obj2.fit(X=X_train, y=y_train)

# model2 평가
y_pred2 =model2.predict(X=X_test)
acc2 = accuracy_score(y_test,y_pred2)
print('accuracy =', acc2)
# accuracy = 0.9769141977683724
 

#################################################
### Grid Search
#################################################
# 최적 parameter를 탐색 : 가장 정확도가 높은 parameter를 찾는 법

# 로직구현
'''
Grid Search : 최적의 매개변수(hyper parameters)를 찾는 방법, model 튜닝
'''
# C, gamma 파라미터
params = [0.001, 0.01, 0.1, 1, 10, 100] # 10e-3 ~ 10e+2
best_score =0        # 최고 분류정확도
best_parameters = {} # 최적 파라미터

for kernel in ['rbf', 'linear']: # kernel 파라미터
    for gamma in params:         # gammas 파라미터
        for C in params:         # cost 파라미터
            #obj = SVC(C=1.0, kernel=kernel, gamma=gamma) # model object
            model = obj.fit(X=X_train, y=y_train)   # 학습
            
            score = model.score(X=X_test, y=y_test) # 평가점수
            # 최적의 점수와 파라미터 갱신
            if best_score < score :
               best_score = score # 점수 갱신
               best_parameters = {'kernel':kernel, 'C':C, 'gamma':gamma}

print('best_score : ', best_score)
# 비선형 svm 모델 : best_score :  0.9782608695652174


print('best_parameters : ', best_parameters)
# 비선형 svm 모델 -> best_parameters : {'kernel': 'rbf', 'C': 0.001, 'gamma': 1}

############################################################################# SVM modeling 분석 결과 ##########################################################################
# 비선형 best_parameters 적용 : model생성
obj = SVC(C=0.001, kernel='rbf', gamma= 1)
model = obj.fit(X=X_train, y=y_train)

train_score = model.score(X=X_train, y=y_train)
test_score = model.score(X=X_test, y=y_test)
print(train_score) # 0.946075197889182
print(test_score)  # 0.9424778761061947

# model 평가
y_pred = model.predict(X = X_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy = ',acc)
# accuracy =  0.9424778761061947

# 혼돈하다..
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)
'''
0[[   0  299]  -> 비중복 민원 예측..0%?
1 [   0 4899]] -> 중복 민원 예측 100%
'''

# 이상 SVM모델 분석 결과였습니다.

