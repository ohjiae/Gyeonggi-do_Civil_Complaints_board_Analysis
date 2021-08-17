# -*- coding: utf-8 -*-
"""
Dicision_Tree.py

결정 트리 모델
"""

# 0. package load
import pandas as pd # csv file
import numpy as np
import string       # texts 전처리
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저

from sklearn.tree import DecisionTreeClassifier # model
from sklearn.model_selection import train_test_split # dataset split
from sklearn.tree import plot_tree, export_text # tree 시각화
from sklearn.metrics import confusion_matrix # 평가


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


"""
chap02
step00_Decision_Tree.py
MODEL CASE32337
1. SVM           - 지은님
2. Naive Baise   - 지애님
3. Decision Tree - 다현님
# 결정 트리로 0(일반민원), 1(중복민원) 분류
"""


# 5. model1: 중요변수='gini', max_depth=3
# train test split
x_train, x_test, y_train, y_test = train_test_split(
    tit_vectorizer, sep, test_size=0.3, random_state=123)

# 1) model 생성
obj1 = DecisionTreeClassifier(criterion='gini',
                             max_depth=3,
                             min_samples_split=2,
                             random_state=123)

model1 = obj1.fit(X=x_train, y=y_train)

# 2) 시각화
tree_text = export_text(model1)
plot_tree(model1)

import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(figsize = (10,10))
plot_confusion_matrix(model1, x_test, y_test, cmap=plt.cm.Blues, ax = ax)  

# 6. model2: 중요변수='gini', max_depth=None
# 1) model 생성
obj2 = DecisionTreeClassifier(criterion='gini',
                             max_depth=None,
                             min_samples_split=2,
                             random_state=123)

model2 = obj2.fit(X=x_train, y=y_train)

# 2) 시각화
tree_text = export_text(model2)
plot_tree(model2)

fig, ax = plt.subplots(figsize = (10,10))
plot_confusion_matrix(model2, x_test, y_test, cmap=plt.cm.Blues, ax = ax)  

# 7. model3: 중요변수='entropy', max_depth=3
# 1) model 생성
obj3 = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3,
                              random_state=123)


model3 = obj3.fit(X=x_train, y=y_train)

# 2) 시각화
tree_text = export_text(model3)
plot_tree(model3)

fig, ax = plt.subplots(figsize = (10,10))
plot_confusion_matrix(model3, x_test, y_test, cmap=plt.cm.Blues, ax = ax)  


# 7. model 평가
# 1) model1
y_pred1 = model1.predict(x_test)

# confusion_matrix
con_mat1 = confusion_matrix(y_test, y_pred1)
print(con_mat1)
'''
[[  61  238]
 [  15 4884]]
'''

# accuracy_score
train_score1 = model1.score(X=x_train, y=y_train)
test_score1 = model1.score(X=x_test, y=y_test)

print(train_score1) # 0.9509399736147758
print(test_score1) # 0.9513274336283186

# 2) model2
y_pred2 = model2.predict(x_test)

# confusion_matrix
con_mat2 = confusion_matrix(y_test, y_pred2)
print(con_mat2)
'''
[[ 167  132]
 [  29 4870]]
'''

# accuracy_score
train_score2 = model2.score(X=x_train, y=y_train)
test_score2 = model2.score(X=x_test, y=y_test)

print(train_score2) # 0.997608839050132
print(test_score2) # 0.9690265486725663

# 3) model3
y_pred3 = model3.predict(x_test)

# confusion_matrix
con_mat3 = confusion_matrix(y_test, y_pred3)
print(con_mat3)
'''
[[   0  299]
 [   0 4899]]
'''

# accuracy_score
train_score3 = model3.score(X=x_train, y=y_train)
test_score3 = model3.score(X=x_test, y=y_test)

print(train_score3) # 0.9463225593667546
print(test_score3) # 0.9424778761061947
