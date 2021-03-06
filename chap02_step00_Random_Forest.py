# ##################################################################### 기존 전처리 과정 ##############################################################
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




########################################################################### RandomForest ######################################################################################
"""
step01_RandomForest.py
"""

from sklearn.ensemble import RandomForestClassifier # model - 분류트리
from sklearn.datasets import load_wine # dataset
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. dataset load
data = pd.read_csv(path +"sep_crawling_data_17326.csv", encoding='CP949')
sep = data['sep']

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, sep, test_size=0.3, random_state=123)

X_train.shape # (12128, 3560)
X_test.shape  # (5198, 3560) 


# 2. model생성
help(RandomForestClassifier)
'''
n_estimators=100 : tree개수 (D)
criterion='gini  : 중요변수 선정 기준
max_depth=None
min_samples_split=2
'''

obj = RandomForestClassifier()          # 위의 기본값을 토대로 obj 생성
model = obj.fit(X=X_train, y=y_train)   # train dataset 적용


# model 평가
y_pred = model.predict(X=X_test)

con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)
'''
[[ 122  177]
 [  10 4889]]
'''
# 특이도 
4889/(177+4889) # 0.9650611922621397
#recall
122/(122+10) #0.92424242424

accuracy_score(y_test, y_pred) # 0.9640246248557137
print(classification_report(y_test, y_pred))
'''
            precision    recall  f1-score   support

           0       0.92      0.41      0.57       299
           1       0.97      1.00      0.98      4899

    accuracy                           0.96      5198
   macro avg       0.94      0.70      0.77      5198
weighted avg       0.96      0.96      0.96      5198
'''

# 시각화
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(figsize = (10,10))
plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Greens, ax = ax)  


