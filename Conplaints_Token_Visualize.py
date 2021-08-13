# -*- coding: utf-8 -*-

"""
Conplaints_Token_Visualize.py

문장 토큰 시각화 - 민원별 추천부서
"""

# 0. package load
# text 처리
import pandas as pd # csv file
import numpy as np
import string       # texts 전처리
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt # 시각화
from matplotlib import font_manager, rc # 한글 깨짐 현상 처리


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

num = [9338, 8343, 9684, 3489, 3588, 1020, 6137, 10015, 9734, 5658,]
#num = np.random.randint(len(titles), size = 10)
tit_num = titles[num]

print(tit_num)


# 2. tit_num 전처리

# 1) 전처리 함수 -> [text_sample.txt] 참고
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
# tit_num 전처리
tit_num = text_prepro(tit_num)
print(tit_num)


# 3. 불용어 제거 - Okt 함수 이용
# 1) 불용어 사전 - https://www.ranks.nl/stopwords/korean
korean_stopwords = path + "korean_stopwords.txt"

with open(korean_stopwords, encoding='utf8') as f : 
    stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]
    
# 2) 불용어 제거
okt = Okt()

tit_num_re = []  # 전처리 완료된 tit_num


# titles 불용어 제거
for sentence in tit_num:  
  tmp1 = okt.morphs(sentence)
  tit_tokenized = []
  
  token_tot = ""    
  for token in tmp1:      
      if not token in stopwords:
          tit_tokenized.append(token)
          token = token + " "
          token_tot += token

  tit_num_re.append(token_tot)

print(tit_num_re)


# 5. text vectorizing(tf-idf)
# 1) 객체 생성
tfidf_vectorizer = TfidfVectorizer()           

# 2) titles 문장 벡터화 진행
# 문장 벡터화 진행
tit_vectorizer = tfidf_vectorizer.fit_transform(tit_num_re)

# 각 단어 벡터화 진행
tit_num_word = tfidf_vectorizer.get_feature_names()  
# 각 단어 벡터값
tit_idf = tfidf_vectorizer.idf_

# 단어, IDF 값 매칭 리스트
tit_idf_list = dict(zip(tit_num_word, tit_idf)) 
# 단어와 부여된 정수값 확인
tit_num_index = tfidf_vectorizer.vocabulary_

print(tit_num_index)


#부서별 키워드 정의
top9_dept = {
    '도로정책과' :['계획','다리','일산대교','착공','변경','이전','교통','통행료','한강','대교','도로'],
    '철도정책과' :['교통','북부','연장','착공','유치','신분당선','신설','면제','확정','지하철','호선','철도','노선', '통일로', '트램'],
    '버스정책과' :['교통','노선','변경','이전','교통카드','교통비','청소년','버스','대중교통','요금'],
    '철도건설과' :['교통','변경','이전','착공','원안','위치','호선','역','지하철','철도','노선', '트램'],
    '신도시추진단' :['시설','유치','이전','주택','지역','원안','위치','신도시', '하수'],
    '총무과' :['이재명','도지사','도지사님','지사님'],
    '감염병관리과' : ['코로나', '검사'],
    '신도시기획과' : ['병합', '부지', '폐기물', '시설', '위치'],
    '질병정책과' : ['코로나', '검사', '백신', '감염']
}

#부서별 키워드를 각 부서별로 하나의 리스트로 통합
top9 = []
for i in top9_dept.values():
    tmp = []
    j_tot = ""
    for j in i:
        tmp.append(i)
        j = j + " "
        j_tot += j
    top9.append(j_tot)

    
#본 파일(17000개)의 tfidf vectorizing    
tit_vectorizer = tfidf_vectorizer.fit_transform(tit_num_re)
tit_td = tit_vectorizer.toarray()

#부서별 키워드의 tfidf vectorizing
sm = tfidf_vectorizer.transform(top9)
sm = sm.toarray()

#부서별 키워드 vs 본 파일 title 간의 유사도 측정
test_sim = linear_kernel(sm, tit_td)

print(test_sim)

df = pd.DataFrame(test_sim, index = top9_dept.keys())
tot_sim = []

def top2_dept(df) : 
    for i in range(10) : 
        t = df.sort_values([i], ascending = False)
    
        if t.iloc[0,i] <= 0.03 : 
            print(i+1, '번째 - 해당 부서 없음')
            
        else : 
            top = t.head(2)
            dept = top.index
            result = dict(zip(list(dept), [t.iloc[0,i], t.iloc[1,i]]))
                
            print(i+1, '번째 -', result)
            tot_sim.append(result)

top2_dept(df)

tot_sim = pd.DataFrame(tot_sim)
tot_sim = tot_sim.fillna(0)

font_path = 'C:/Windows/Fonts/malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

tot_sim.index

tot_sim.plot(kind = 'barh', stacked = True, title = "민원별 추천부서")
plt.xlabel("유사도")
plt.ylabel("민원")

y1 = tot_sim['신도시추진단']
y2 = tot_sim['도로정책과']
y3 = tot_sim['신도시기획과']
y4 = tot_sim['철도정책과']
'''
for i, v in enumerate(round(y1,2)) : 
    if y1[i] == 0 : 
        pass
    else : 
        plt.text(v-0.055, i-0.15, str(v), fontsize = 10)
        
for i, v in enumerate(round(y2,2)) : 
    if y2[i] == 0 : 
        pass
    else : 
        plt.text(v-0.055, i-0.15, str(v), fontsize = 10)
        
for i, v in enumerate(round(y3,2)) : 
    if y3[i] == 0 : 
        pass
    else : 
        plt.text(v+0.2, i-0.15, str(v), fontsize = 10)
        
for i, v in enumerate(round(y4,2)) : 
    if y4[i] == 0 : 
        pass
    else : 
        plt.text(v-0.055, i-0.15, str(v), fontsize = 10)
'''
plt.show()
