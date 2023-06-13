#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1.SORU A)


# In[60]:


import pandas as pd

data = {'personal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Maas': ['normal', 'yuksek', 'dusuk', 'normal', 'normal', 'dusuk', 'yuksek', 'yuksek', 'normal', 'dusuk', 'yuksek'],
    'deneyim': ['orta', 'iyi', 'yok', 'iyi', 'orta', 'yok', 'iyi', 'orta', 'yok', 'iyi', 'orta'],
    'gorev': ['uzman', 'yonetici', 'uzman', 'yonetici', 'uzman', 'uzman', 'yonetici', 'uzman', 'yonetici', 'uzman', 'yonetici'],
    'memnun': ['evet', 'hayir', 'evet', 'hayir', 'evet', 'evet', 'hayir', 'evet', 'hayir', 'evet', 'hayir']}

df = pd.DataFrame(data)
print(df)


# In[ ]:


1.SORU B)


# In[17]:


import pandas as pd


data = {'': ['Maaşı normal olanlar', 'Maaşı düşük olanlar', 'Maaşı yüksek olanlar'],
        'Sol tarafta': [50, 14, 36],
        'Sağ tarafta': [22, 7, 23]}


example_counts = {'Maaşı normal olanlar': {'Sol tarafta': 0, 'Sağ tarafta': 0},
                  'Maaşı düşük olanlar': {'Sol tarafta': 0, 'Sağ tarafta': 0},
                  'Maaşı yüksek olanlar': {'Sol tarafta': 0, 'Sağ tarafta': 0}}


for i in range(len(data)):
    
    if i == 0:
        example_counts['Maaşı normal olanlar']['Sol tarafta'] = data['Sol tarafta'][i]
        example_counts['Maaşı düşük olanlar']['Sol tarafta'] = data['Sol tarafta'][i + 1]
        example_counts['Maaşı yüksek olanlar']['Sol tarafta'] = data['Sol tarafta'][i + 2]
    elif i == 1:
        example_counts['Maaşı normal olanlar']['Sol tarafta'] = data['Sol tarafta'][i - 1] + data['Sol tarafta'][i + 1]
        example_counts['Maaşı düşük olanlar']['Sol tarafta'] = data['Sol tarafta'][i]
        example_counts['Maaşı yüksek olanlar']['Sol tarafta'] = data['Sol tarafta'][i + 1]
    elif i == 2:
        example_counts['Maaşı normal olanlar']['Sol tarafta'] = data['Sol tarafta'][i - 2] + data['Sol tarafta'][i - 1]
        example_counts['Maaşı düşük olanlar']['Sol tarafta'] = data['Sol tarafta'][i - 1]
        example_counts['Maaşı yüksek olanlar']['Sol tarafta'] = data['Sol tarafta'][i]

    
    if i == 0:
        example_counts['Maaşı normal olanlar']['Sağ tarafta'] = data['Sağ tarafta'][i]
        example_counts['Maaşı düşük olanlar']['Sağ tarafta'] = data['Sağ tarafta'][i + 1]
        example_counts['Maaşı yüksek olanlar']['Sağ tarafta'] = data['Sağ tarafta'][i + 2]
    elif i == 1:
        example_counts['Maaşı normal olanlar']['Sağ tarafta'] = data['Sağ tarafta'][i - 1] + data['Sağ tarafta'][i + 1]
        example_counts['Maaşı düşük olanlar']['Sağ tarafta'] = data['Sağ tarafta'][i]
        example_counts['Maaşı yüksek olanlar']['Sağ tarafta'] = data['Sağ tarafta'][i + 1]
    elif i == 2:
        example_counts['Maaşı normal olanlar']['Sağ tarafta'] = data['Sağ tarafta'][i - 2] + data['Sağ tarafta'][i - 1]
        example_counts['Maaşı düşük olanlar']['Sağ tarafta'] = data['Sağ tarafta'][i - 1]
        example_counts['Maaşı yüksek olanlar']['Sağ tarafta'] = data['Sağ tarafta'][i]


df = pd.DataFrame(example_counts)


print(df)


# In[ ]:


1.SORU C)


# In[62]:


import pandas as pd


data = {'Maas': ['Düşük', 'normal', 'Yüksek'],
        'Düşük Ekonomik Durum': [6, 6, 1],
        'normal Ekonomik Durum': [3, 5, 2],
        'Yüksek Ekonomik Durum': [1, 2, 5]}
df = pd.DataFrame(data).set_index('Maas')


def twoing(df, feature, target):
    n = df.sum().sum()
    ni_ = df.sum(axis=1)
    nj_ = df.sum(axis=0)
    ni = ni_.loc[feature]
    nj = nj_.loc[target]
    a = df.loc[feature, target]
    b = ni - a
    c = nj - a
    d = n - a - b - c
    p_t = (a + b) / n
    p_f = (c + d) / n
    p_ft = a / (a + b)
    p_ff = c / (c + d)
    delta_p = abs(p_ft - p_ff)
    delta_p0 = abs(p_t - p_f)
    return delta_p * delta_p0

twoing_scores = {}
for feature in df.index:
    for target in df.columns:
        score = twoing(df, feature, target)
        twoing_scores[(feature, target)] = score

top_two = sorted(twoing_scores.items(), key=lambda x: x[1], reverse=True)[:2]


left_data = df.loc[[top_two[0][0][0], top_two[1][0][0]]]
right_data = df.drop([top_two[0][0][0], top_two[1][0][0]])

left_prob = left_data.sum(axis=0) / left_data.sum().sum()
right_prob = right_data.sum(axis=0) / right_data.sum().sum()

left_df = pd.DataFrame(left_prob).transpose().rename_axis('Sol Durum').rename(columns={'Düşük Ekonomik Durum': 'Sol Olasılık'})
right_df = pd.DataFrame(right_prob).transpose().rename_axis('Sağ Durum').rename(columns={'Düşük Ekonomik Durum': 'Sağ Olasılık'})


print("Sol Durum Olasılıkları:")
print(left_df)

print("\nSağ Durum Olasılıkları:")
print(right_df)


# In[ ]:


2.SORU A)


# In[36]:


import pandas as pd


diabetes = load_diabetes()


df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target


age_labels = ["genç", "orta yaşlı", "yaşlı"]
cut_points = [0, 30, 40, float("inf")]
df["age_group"] = pd.cut(df["age"], bins=cut_points, labels=age_labels)


df["target"].replace({1: "1", 0: "0"}, inplace=True)


print(df.head())


# In[ ]:


2.SORU B)


# In[50]:


p_genc_bilgisayar_almak = 0.6
p_medium_income_bilgisayar_almak = 0.4
p_student_yes_bilgisayar_almak = 0.3
p_credit_fair_bilgisayar_almak = 0.5

p_genc_bilgisayar_almamak = 0.2
p_medium_income_bilgisayar_almamak = 0.6
p_student_yes_bilgisayar_almamak = 0.7
p_credit_fair_bilgisayar_almamak = 0.3


p_x_bilgisayar_almak = p_genc_bilgisayar_almak * p_medium_income_bilgisayar_almak * p_student_yes_bilgisayar_almak * p_credit_fair_bilgisayar_almak


p_x_bilgisayar_almamak = p_genc_bilgisayar_almamak * p_medium_income_bilgisayar_almamak * p_student_yes_bilgisayar_almamak * p_credit_fair_bilgisayar_almamak


p_x = p_x_bilgisayar_almak * 0.5 + p_x_bilgisayar_almamak * 0.5


p_bilgisayar_alacak = p_x_bilgisayar_almak * 0.5 / p_x
p_bilgisayar_almayacak = p_x_bilgisayar_almamak * 0.5 / p_x

print('Kişinin bilgisayar alıp alma olasılığı:', p_bilgisayar_alacak)
print('Kişinin bilgisayar almama olasılığı:', p_bilgisayar_almayacak)


# In[ ]:


3.SORU A)


# In[51]:


import pandas as pd

data = {'X': [0, 1, 2, 3, 4],
        'Y': [1, 3, 2, 5, 4]}

df = pd.DataFrame(data)

print(df)
print('X koordinatları:', df['X'].values)
print('Y koordinatları:', df['Y'].values)


# In[ ]:


3.SORU B)


# In[57]:


import pandas as pd
import math


data = {'X': [0, 1, 2, 3, 4], 'Y': [1, 3, 2, 5, 4]}
df = pd.DataFrame(data)


new_data = {'X': [7], 'Y': [3]}
new_df = pd.DataFrame(new_data)
df = df.append(new_df, ignore_index=True)


distances = []
for i in range(len(df)):
    dist = math.sqrt((df['X'][i]-7)**2 + (df['Y'][i]-3)**2)
    distances.append(dist)
df['Distance'] = distances


df = df.sort_values(by=['Distance'])
df['Rank'] = range(1, len(df)+1)


df.loc[4] = [7, 3, distances[4-1], 1]
df['Rank'] = range(1, len(df)+1)


print(df)


# In[ ]:


5.SORU


# In[73]:


import numpy as np


X = np.array([[4.2, 2.1, 1.5, 0.2],
              [6.3, 3.3, 4.5, 1.5],
              [5.1, 1.9, 1.4, 0.3],
              [6.2, 2.4, 4.3, 1.3],
              [5.4, 2.2, 1.3, 0.4],
              [4.3, 2.1, 1.6, 0.3]])


Y = np.array([0, 1, 0, 1, 0, 0])


new = np.array([5, 3.4, 1.5, 0.2])


X_transpose = np.transpose(X)
X_transpose_X = np.dot(X_transpose, X)
X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
X_transpose_Y = np.dot(X_transpose, Y)

beta_hat = np.dot(X_transpose_X_inverse, X_transpose_Y)


print('Doğrusal Regresyon Katsayıları:', beta_hat)


y_pred = np.dot(beta_hat, new)

if y_pred > 0.5:
    print("Yeni örnek sınıf B'ye aittir.")
else:
    print("Yeni örnek sınıf A'ya aittir.")

