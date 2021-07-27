#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_project

1. data crawling


"""

import urllib.request
import urllib.parse
import requests
from bs4 import BeautifulSoup

res = requests.get("http://google.com")
res.raise_for_status()
print('웹 연결 성공...')
#문제 생기면 여기서 바로 끝남! 쓸데없는 시간 낭비를 줄여줌


#tnwjd
