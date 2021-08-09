#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chap02_step02_Dict_department
1. 부서(Key)와 관련 키워드(Value)를 가진 딕셔너리 만들기
2. 키워드(Value)를 통해 해당 부서(Key)를 찾는 함수 만들기
"""


top5_dept = {
    '철도정책과' :,
    '버스정책과' :,
    '신도시기획과' :,
    '총무과' :,
    '신도시추진과' :,
}

def find_dept(dict, key_value):
  return list(key for key, value in dict.items() if value == key_value)


print(find_dept(top5_dept,'키워드값(value)'))