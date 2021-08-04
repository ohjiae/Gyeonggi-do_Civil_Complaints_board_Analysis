# -*- coding: utf-8 -*-

"""
- 희소행렬 + DNN model
<작업절차>
1. csv file load
2. texts와 label 전처리
3. num_words = 4000 제한
4. Sparse matrix : features 
5. train/ test split
6. DNN model
"""

# 0. package load
# text 처리
import pandas as pd # csv file
import numpy as np  # list -> numpy
import string       # texts 전처리

from tensorflow.keras.preprocessing.text import Tokenizer         # 토큰 생성기
from sklearn.model_selection import train_test_split              # split

# DNN model 생성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. csv file load
#path = 'K:/ITWILL/Final_project/'
path = 'C:/Users/STU-16/Desktop/ITWILL-Final_project-main/'
minwon_data = pd.read_csv(path + 'minwon_crawling4400.csv', header = None)
minwon_data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4401 entries, 0 to 4400
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   0       4401 non-null   object
 1   1       1 non-null      object
 2   2       4401 non-null   object
 3   3       4400 non-null   object
'''

titles = minwon_data[0]
contents = minwon_data[1]
replies = minwon_data[2]
sep = minwon_data[3]

print(titles)


# 2. titles, contents, replies 전처리
# 1) sep 전처리
label = [1 if lab == '1' else 0 for lab in sep]
label = np.array(label)

# 2) titles 전처리 -> [text_sample.txt] 참고
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

# 3) 함수 호출
titles = text_prepro(titles)
print(titles)


# 3. num_words = 1000개 제한()
tokenizer = Tokenizer()                 # 1차 : 전체 단어 이용
tokenizer = Tokenizer(num_words = 1000) # 2차 : 1000개의 단어 이용 -> 희소행렬에 영향
tokenizer.fit_on_texts(titles)          # 텍스트 반영 -> token 생성
token = tokenizer.word_index            # 토큰 반환  

print(token) # {'word':고유숫자}


# 4. Sparse matrix: [docs, terms]
x_data = tokenizer.texts_to_matrix(texts=titles, mode='tfidf')


# 5. train/test split
x_train, x_val, y_train, y_val = train_test_split(
    x_data, label, test_size=20)


# 6. DNN model
model = Sequential() # keras model

input_shape = (1000, )

# hidden layer1: w[4401, 64] -> w[1000, 64]
model.add(Dense(units=64, input_shape=input_shape, activation='relu')) # 1층

# hidden layer2: w[64, 32]
model.add(Dense(units=32, activation='relu')) # 2층

# output layer: w[32, 1]
model.add(Dense(units=1, activation='sigmoid')) # 3층

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)              (None, 64)                64064     
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
'''


# 7. model compile: 학습과정 설정(이항분류기)
model.compile(optimizer='adam',
              loss = 'binary_crossentropy', # y: one hot encoding
              metrics=['accuracy'])


# 8. model training: train(80) vs val(20)
model.fit(x=x_train, y=y_train, # 훈련셋
          epochs=5, # 반복학습
          batch_size=512,
          verbose=1, # 출력여부
          validation_data=(x_val, y_val)) # 검증셋


# 9. model evaluation: val dataset
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)
