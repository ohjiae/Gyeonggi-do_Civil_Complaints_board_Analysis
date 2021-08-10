# -*- coding: utf-8 -*-
'''
cosine similarity를 사용한 simple 예측

전체 글의 코사인 유사도를 계산, 각 글과 다른 글 사이의 유사도를 전부 검정한 array를 만든다.

만약 유사도가 일정 수치를 넘어간 글이 2개 이상인 경우 중복글로 판정하도록 한다.

이후, 이를 confusion matrix로 정답율을 체크한다. -> 어느정도 성능은 나옵니다.


추가] 각 부서별 키워드와의 유사도를 비교하여, 민원, 특히 자주 나오는 민원들을 분류할 수 있도록 classification 합니다.

'''

# 0. package load
import pandas as pd # csv file
import numpy as np
import string       # texts 전처리
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저

from sklearn.metrics import confusion_matrix, accuracy_score # 평가


# 1. csv file load
minwon_data = pd.read_csv('sep_crawling_data_17326.csv', encoding = 'CP949')
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
print(titles[:10])

# replies 전처리
replies = text_prepro(replies)
print(replies[:3])


# 3. 불용어 제거 - Okt 함수 이용
# 1) 불용어 사전 - https://www.ranks.nl/stopwords/korean
korean_stopwords = "korean_stopwords.txt"

with open(korean_stopwords, encoding='utf8') as f : 
    stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]
    
# 2) 불용어 제거
okt = Okt()

tit_result = []  # 전처리 완료된 titles
rpl_result = []  # 전처리 완료된 replies


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



# 4. text vectorizing(tf-idf)
# 1) 객체 생성
tfidf_vectorizer = TfidfVectorizer()           

# 2) titles 문장 벡터화 진행
# 문장 벡터화 진행
# 문장을 배열로 만들고, 배열에 따른 유사도를 계산, for문을 통해 원하는 prediction을 만들어 볼 것입니다.

tit_vectorizer = tfidf_vectorizer.fit_transform(tit_result)
tit_td = tit_vectorizer.toarray()

    
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm


cosine_sim = linear_kernel(tit_td, tit_td)

# aa = cosine_sim
# aa = aa.tolist()

pred = []
for i in tqdm(cosine_sim):
    tmp = []
    for j in i:
        if j > 0.60:
            tmp.append(j)
    if len(tmp) > 2:
        pred.append(1)
    else:
        pred.append(0)
    
    #     print(cosine_sim[i][j], end=' ')
    # print()

print(confusion_matrix(sep, pred))

print(accuracy_score(sep, pred))


'''

유사도 > 0.60
[[  822   131]
 [  590 15783]]
0.9583862403324483

유사도 > 0.50
[[  768   185]
 [  411 15962]]
0.9656008311208588

유사도 > 0.40
[[  662   291]
 [  235 16138]]
0.9696410019623687

유사도 > 0.30
[[  447   506]
 [   95 16278]]
0.9653122474893224

'''


top9_dept = {
    '도로정책과' :['계획','다리','일산대교','착공','변경','이전','교통','통행료','한강','대교','도로'],
    '철도정책과' :['교통','북부','연장','착공','유치','신분당선','신설','역','면제','확정','지하철','호선','철도','노선', '통일로', '트램'],
    '버스정책과' :['교통','노선','변경','이전','교통카드','교통비','청소년','버스','대중교통','요금'],
    '철도건설과' :['교통','변경','이전','착공','원안','위치','호선','역','지하철','철도','노선', '트램'],
    '신도시추진단' :['시설','유치','이전','주택','지역','원안','위치','신도시', '하수'],
    '총무과' :['이재명','도지사','도지사님','지사님'],
    '감염병관리과' : ['코로나', '검사'],
    '신도시기획과' : ['병합', '부지', '폐기물', '시설', '위치'],
    '질병정책과' : ['코로나', '검사', '백신', '감염']
}

top9_dept = [['계획','다리','일산대교','착공','변경','이전','교통','통행료','한강','대교','도로'],
    ['교통','북부','연장','착공','유치','신분당선','신설','역','면제','확정','지하철','호선','철도','노선', '통일로', '트램'],
    ['교통','노선','변경','이전','교통카드','교통비','청소년','버스','대중교통','요금'],
    ['교통','변경','이전','착공','원안','위치','호선','역','지하철','철도','노선', '트램'],
    ['시설','유치','이전','주택','지역','원안','위치','신도시', '하수'],
    ['이재명','도지사','도지사님','지사님'],
    ['코로나', '검사'],
    ['병합', '부지', '폐기물', '시설', '위치'],
    ['코로나', '검사', '백신', '감염']]

cosine_sim_dept = linear_kernel(tit_td, top9_dept)
