"""
visualization_dept_pie10.py
"""
import pandas as pd # csv file
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#부서별 키워드 정의
top9_dept = {
    '도로정책과' :['계획','다리','일산대교','착공','변경','이전','교통','통행료','한강','대교','도로'],
    '철도정책과' :['교통','북부','연장','착공','유치','신분당선','신설','면제','확정','지하철','호선','철도','노선', '통일로', '트램'],
    '버스정책과' :['교통','노선','변경','이전','교통카드','교통비','청소년','버스','대중교통','요금'],
    '철도건설과' :['교통','변경','이전','착공','원안','위치','호선','지하철','철도','노선', '트램'],
    '신도시추진단' :['시설','유치','이전','주택','지역','원안','위치','신도시', '하수'],
    '총무과' :['이재명','도지사','도지사님','지사님'],
    '감염병관리과' : ['코로나', '검사'],
    '신도시기획과' : ['병합', '부지', '폐기물', '시설', '위치'],
    '질병정책과' : ['코로나', '검사', '백신', '감염']
}

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
# 한글 깨짐 현상 처리
font_path = 'C:/Windows/Fonts/malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

       
labels = list(top9_dept.keys()) # dict -> list
labels.append('외 61개 부서')
print(labels)
'''
['도로정책과', '철도정책과', '버스정책과', '철도건설과', '신도시추진단', '총무과', '감염병관리과', '신도시기획과', '질병정책과', 'etc']
'''

etc = 100- (4.55 +14.90+ 14.39 +5.56 + 3.79+ + 4.30+ +3.28 + 4.55+ 5.05 )
print(etc) # 39.63%
ratio = [4.55,14.90, 14.39, 5.56, 3.79, 4.30, 4.55, 5.05, 3.28, 39.63]
plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=200)
plt.show()

'''
# 부서 top3 =[철도정책과, 버스정책과, 철도건설과, 질병정책과]
# 철도정책과
- 신분당선 서북부 연장의 조속한 예타 확정과 조기착공 촉구
- 통일로선 추진과 신원역 신설
- 3호선 복선 하남 연장 원안 이행 촉구

# 버스정책과
- 경기도 청소년 교통비 지원 관련 문의
- 버스 시간표 및 노선관련

# 철도건설과
- 동탄 트램-문화디자인밸리 변경 요청
- 지하철 7호선 장암역 이전과 민락역 신설

# 질병정책과
- 코로나10 선별 진료소
- 영유아 코로나 확진자를 위한 지원 지침 제안
- 수도권 국민 코로나 의무검사
'''
