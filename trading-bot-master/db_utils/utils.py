import os
import math
import logging
from datetime import datetime

import pandas as pd
import numpy as np

import keras.backend as K

import pymysql
import db_utils.private_info as pv_info

conn = None
cursor = None

# 데이타 Fetch
# rows = cursor.fetchall()

action_dict = {
    "HOLD":0,
    "BUY":1,
    "Cannot BUY":2,
    "SELL":3,
    "Cannot SELL":4
}

# TODO History 형태로 넘어오면 SQL 쿼리로 변환
# 데이터 삽입
def insert_data(corp_code, data, profit, prefix):
    global conn, cursor

    date = data['date'].date()
    corp = corp_code
    price = int(data['price'])
    volume = int(data['volume'])
    action = int(action_dict[data['action']])
    profit = int(profit)
    prefix = float(prefix)

    print(type(price), type(volume), type(action))

    # SQL문 실행
    # Create a new record
    sql = """INSERT INTO table_logs (trading_date, trading_corp, trading_price, trading_volume, trading_action, trading_profit, trading_prefix)
             VALUES (%s, %s, %s, %s, %s, %s, %s) """
    val = (date, corp, price, volume, action, profit, prefix)
    cursor.execute(sql, val)

# 데이터 조회
# TODO : Trading Bot 늘어나면 코드 개선 필요할 듯
def get_data(corp_code, times=30):
    global conn, cursor

    sql = """SELECT table_logs.*, table_corp.corp_name FROM table_logs INNER JOIN table_corp """+ \
          """ON table_logs.trading_corp=table_corp.corp_id AND table_corp.corp_name="""+"""삼성전자"""+ \
          """LIMIT """+str(times)

    cursor.execute(sql)
    result = cursor.fetchall()

# 연결
def connect_server():
    global conn, cursor

    # MySQL Connection 연결
    conn = pymysql.connect(host=pv_info.host_name, port=pv_info.port_num,
                           user=pv_info.user_name, password=pv_info.user_password,
                           db=pv_info.database, charset='utf8', autocommit=True)

    # Connection 으로부터 Cursor 생성
    cursor = conn.cursor()

# 연결 해제
def disconnect_server():
    conn.close()