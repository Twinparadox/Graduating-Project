import pandas as pd
import numpy as np
import datetime
import time

code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]
# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌
code_df.종목코드 = code_df.종목코드.map('{:06d}'.format)
# 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다.
code_df = code_df[['회사명', '종목코드']]
# 한글로된 컬럼명을 영어로 바꿔준다.
code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})
print(code_df)

# 종목 이름을 입력하면 종목에 해당하는 코드를 불러와
# 네이버 금융(http://finance.naver.com)에 넣어줌
def get_url(item_name, code_df):
    code = code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
    code = code[1:]
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)

    print("요청 URL = {}".format(url))
    return url

# 보조지표
# 네이버는 15, 5, 3을, 일반 증권사는 5, 3, 3 사용
# 일자(n,m,t)에 따른 Stochastic(KDJ)의 값을 구하기 위한 함수
# 스토캐스틱 20 이하에서 %K선이 %D선을 상향 돌파하면 골든크로스라고 하여 매수 관점으로 접근할 수 있고,
# 스토캐스틱 80 이상에서 %K선이 %D선을 하향 돌파하면 데드크로스라고 하여 매도 관점
def get_stochastic(df, n=5, m=3, t=3):
    # 입력받은 값이 dataframe이라는 것을 정의해줌
    df = pd.DataFrame(df)

    # n일중 최고가
    ndays_high = df.high.rolling(window=n, min_periods=1).max()
    # n일중 최저가
    ndays_low = df.low.rolling(window=n, min_periods=1).min()

    # Fast%K 계산
    kdj_k = ((df.close - ndays_low) / (ndays_high - ndays_low)) * 100
    # Fast%D (=Slow%K) 계산
    kdj_d = kdj_k.ewm(span=m).mean()
    # Slow%D 계산
    kdj_j = kdj_d.ewm(span=t).mean()

    # dataframe에 컬럼 추가
    df = df.assign(kdj_k=kdj_k, kdj_d=kdj_d, kdj_j=kdj_j)

    return df
# MACD
# 이동평균선이 수렴과 발산을 반복한다는 원리를 이용해 단기이동평균선과 장기이동평균선 사이의 관계를 보여줌
# MACD : 단기이동평균과 장기이동평균선의 차이값
# Signal : MACD의 9일 이동평균값
# Oscillator : MACD값과 Signal값의 차이

def get_macd(df, short=12, long=26, t=9):
    df = pd.DataFrame(df)

    # 단기(12) EMA(지수이동평균)
    ma_12 = df.close.ewm(span=short).mean()
    # 장기(26) EMA
    ma_26 = df.close.ewm(span=long).mean()

    # MACD
    macd = ma_12 - ma_26
    # Signal
    macds = macd.ewm(span=t).mean()
    # Oscillator
    macdo = macd - macds

    df = df.assign(macd=macd, macds=macds, macdo=macdo)

    return df

def get_cci(df, n=20):
    df = pd.DataFrame(df)

    # (고가 + 저가 + 종가) / 3
    M = (df.high + df.low + df.close)/3
    # M의 n일 단순이동평균
    m = M.rolling(n).mean()
    # |M-m|의 n일 단순이동평균
    d = abs(M-m).rolling(n).mean()

    cci = (M-m) / (d*0.015)

    df = df.assign(cci=cci)

    return df

def calculate_diff(df):
    df = pd.DataFrame(df)

    diff = df.close.diff()
    df['diff'] = diff

    D = np.where(df['diff']<0, df['diff'], 0)
    U = np.where(df['diff']>0, df['diff'], 0)

    df = df.assign(D=D, U=U)

    return df

def get_rsi(df, n=6):
    df = pd.DataFrame(df)

    AU = df.U.rolling(n).mean()
    AD = df.D.rolling(n).mean()
    rsi = np.where(AU+AD==0, 0, AU/(AU+AD))

    df = df.assign(rsi=rsi)

    return df

# 단순 이동평균
def get_sma(df):
    df = pd.DataFrame(df)

    sma5 = df.close.rolling(5).mean()
    sma10 = df.close.rolling(10).mean()
    sma20 = df.close.rolling(20).mean()

    df = df.assign(sma5=sma5, sma10=sma10, sma20=sma20)

    return df
# 지수 이동평균
def get_ema(df):
    df = pd.DataFrame(df)

    ema5 = df.close.ewm(span=5).mean()
    ema10 = df.close.ewm(span=10).mean()
    ema20 = df.close.ewm(span=20).mean()

    df = df.assign(ema5=ema5, ema10=ema10, ema20=ema20)

    return df

# 가중 이동평균
def wighted_mean(weight_array):
    def inner(x):
        return (weight_array * x).mean()
    return inner

def get_wma(df):
    df = pd.DataFrame(df)

    weights = np.arange(1,6)
    wma5 = df.close.rolling(5).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

    weights = np.arange(1,11)
    wma10 = df.close.rolling(10).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

    weights = np.arange(1,21)
    wma20 = df.close.rolling(20).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

    df = df.assign(wma5=wma5, wma10=wma10, wma20=wma20)

    return df


item_name_list = ['삼성전자', 'NAVER', '현대자동차', '포스코', '카카오', 'SK텔레콤', '아모레퍼시픽', '신한지주', '넷마블',
                  '셀트리온', 'SK하이닉스', 'KB금융', 'LG생활건강', '기아자동차', '케이티', 'LG화학',
                  '롯데케미칼', '삼성SDI', '엔씨소프트', '대한항공', '유한양행', '안랩', '오뚜기', '이마트',
                  '삼성생명', '현대모비스', '한국전력공사']

for item_name in item_name_list:
    print('read ' + item_name + ' data')
    url = get_url(item_name, code_df)

    # 일자 데이터를 담을 df라는 DataFrame 정의
    df = pd.DataFrame()

    # 1페이지에서 20페이지의 데이터만 가져오기
    for page in range(1, 265):
        pg_url = '{url}&page={page}'.format(url=url, page=page)
        df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
        time.sleep(0.05)
        # print("PAGE = {0} Done".format(page))

    # df.dropna()를 이용해 결측값 있는 행 제거
    df = df.dropna()

    # 한글로 된 컬럼명을 영어로 바꿔줌
    df = df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff',
                            '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})

    # 데이터의 타입을 int형으로 바꿔줌
    df[['close', 'diff', 'open', 'high', 'low', 'volume']] \
        = df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)

    # 컬럼명 'date'의 타입을 date로 바꿔줌
    df['date'] = pd.to_datetime(df['date'])

    # 일자(date)를 기준으로 오름차순 정렬
    df = df.sort_values(by=['date'], ascending=True)

    df = get_stochastic(df)
    df = get_macd(df)
    df = get_cci(df)
    df = get_sma(df)
    df = get_ema(df)
    df = get_wma(df)
    df = calculate_diff(df)
    df = get_rsi(df)
    df = df.dropna()

    df.to_csv('stockdata/' + item_name + '.csv', mode='w', index=False)
    print('saved ' + item_name + '.csv')
