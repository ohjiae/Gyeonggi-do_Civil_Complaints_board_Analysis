import pandas as pd # csv file
import numpy as np  # list -> numpy
import string       # texts 전처리


from konlpy.tag import Mecab #형태소 분석

# 단어 빈도수 확인, 코사인 유사도

from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이저

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity # 코사인 유사도


# 파일 읽어오기 
# 1. csv file load
path = '/Users/aegohc/ITWILL/final_project/'
minwon_data = pd.read_csv(path + 'minwon_crawling4400.csv', header = None)
minwon_data.info()


titles = minwon_data[0]
contents = minwon_data[1]
replies = minwon_data[2]
sep = minwon_data[3]


# 2. 한글 토큰화
'''
def mecab_tnouns(titles):
    mecab = Mecab()
    titles_ndata = ' '.join(mecab.nouns(titles))
    return titles_ndata


def mecab_tmorphs(titles):
    mecab = Mecab()
    titles_ndata = ' '.join(mecab.morphs(titles))
    return titles_ndata
'''

## 1) titles 전처리 -> [text_sample.txt] 참고
def text_prepro(titles):
    # Lower case : 소문자
    titles = [x.lower() for x in titles]
    # Remove punctuation : 문장부호 제거
    titles = [''.join(c for c in x if c not in string.punctuation) for x in titles]
    # 특수문자 제거 필요
    
    
    return titles

## 2) 불용어 제거 #'주세요','해주세요', '드립니다','합니다','입니다'
stopwords=['주세요','해주세요', '드립니다','합니다','입니다','을','의','가','이','은','들','는','좀','잘','과','도','를','으로','자','에','에서','와','한','하다']

## 3) Mecab으로 형태소 토큰화하기
mecab = Mecab()

tit_res = []
for sentence in titles:
  tmp = []
  tmp = mecab.morphs(sentence)
  
  tit_tokenized = []
  for token in tmp:
    if not token in stopwords:
      tit_tokenized.append(token)


  tit_res.append(tit_tokenized)

print(tit_res)

len(tit_res)

# 3. 벡터화.
# TF-IDF를 자동계산해주는 라이브러리 사용
## 1) 정수 인코딩 (Integer Encoding)
# 데이터 학습 : TfidfVectorizer().fit(i) ㅇㅕ기 i 자리에 리스트 1개만 들어가야하는데, tit_res는 여러 리스트가 모여있다... 어떻게해야하지.. Forㅁㅜㄴ..?
'''
for k in tit_res :
    k = 0
    if k < 4401:
        tfidf_t = TfidfVectorizer().fit(tit_res[k])
        print(tfidf_t.transform(tit_res[k]).toarray())
        tit_res.append(tfidf_t)
        k += 1
    else:
        tfidf_t = TfidfVectorizer().fit(tit_res[k])
        print(tfidf_t.transform(tit_res[k]).toarray())
        tit_res.append(tfidf_t)
        break()
'''

# 한 리스트는 되는데...전체로 하면 ㅠ # AttributeError: 'list' object has no attribute 'lower'
tfidf_t = TfidfVectorizer().fit(tit_res[2])
print(tfidf_t.transform(tit_res[2]).toarray())
print(tfidf_t.vocabulary_)

tfidf_matrix = tfidf_t.fit_transform(tit_res[2]).toarray()
cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]) #돌아가는데 유사도가 안나옴^^>.

'''
## 문장 벡터화 진행
tfidf_t = TfidfVectorizer()
tfidf_matrix = tfidf_t.fit_transform(tit_res).toarray()
idf = tfidf_v.idf_
print(dict(zip(tfidf_t.get_feature_names(), idf)))
cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

## 함수 인자로는 행렬과 배열만 가능하므로 np.array로 변환
titles_array = np.array(titles) 


print(tfidf_v.transform(titles).toarray()) # titles 안의 각 단어 빈도 수를 기록.
words_index_t = tfidf_v.vocabulary_ # 각 단어의 인덱스 어떻게 부여되었는지 프린트


print(tfidf_matrix.shape)
#(4401, 1562) #4401의 게시글에서 1562개의 단어가 쓰임


# 코사인 유사도 구하기
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print(cosine_sim)
'''
