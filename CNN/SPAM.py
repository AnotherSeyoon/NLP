import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename = "spam.csv")
data = pd.read_csv('spam.csv', encoding = 'latin-1')

# print('총 샘플의 수 :', len(data))

# print(data[:5])

# 불필요한 열 제거 및 레이블 정수(0, 1)로 변경
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham',  'spam'], [0, 1])

# print(data[:5])

# print(data['v2'].nunique(), data['v1'].nunique())

data.drop_duplicates(subset = ['v2'], inplace = True) # v2 열에서 중복제거

data['v1'].value_counts().plot(kind = 'bar')
# plt.show()

# print(data.groupby('v1').size().reset_index(name = 'count'))

X_data = data['v2']
y_data = data['v1']
print('메일 본문의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))