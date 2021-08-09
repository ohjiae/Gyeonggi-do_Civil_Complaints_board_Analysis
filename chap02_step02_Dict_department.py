"""
chap02_step02_Dict_department
1. 부서(Key)와 관련 키워드(Value)를 가진 딕셔너리 만들기
2. 키워드(Value)를 통해 해당 부서(Key)를 찾는 함수 만들기
"""


top5_dept = {
    '도로정책과' :['계획','다리','일산대교','착공','변경','이전','교통','면제','통행료','한강','대교','도로'],
    '철도정책과' :['교통','북부','연장','착공','유치','신분당선','신설','역','면제','확정','지하철','호선','철도','노선'],
    '버스정책과' :['교통','노선','변경','이전','교통카드','교통비','청소년','버스','대중교통','요금'],
    '철도건설과' :['교통','변경','이전','착공','원안','위치','호선','역','지하철','철도','노선'],
    '신도시추진단' :['시설','유치','이전','주택','지역','원안','위치','신도시'],
    '총무과' :['이재명','도지사','도지사님','지사님']
    
}


def find_dept(keyword):
    print(list(value for value in dict.values() if value == keyword))
    
print(find_dept('다리'))

