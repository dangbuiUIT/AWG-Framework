from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import random
import math
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Flatten, Conv1D, Dropout, BatchNormalization, Embedding
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from keras.losses import BinaryCrossentropy
from keras.optimizers import SGD, Adam
from urllib.parse import urlparse
import idna

def parse_url(url):
    # Sử dụng hàm urlparse để phân tích URL thành các thành phần
    parsed_url = urlparse(url)

    # Lấy ra các trường thông tin từ URL
    protocol = parsed_url.scheme  # Protocol (http, https, ftp, etc.)
    sub_domain = parsed_url.hostname.split('.')[0]  # Sub-domain name
    domain_name = '.'.join(parsed_url.hostname.split('.')[1:-1])  # Domain name
    domain_suffix = parsed_url.hostname.split('.')[-1]  # Domain suffix (com, org, vn, etc.)
    path = parsed_url.path  # URL path

    return protocol, sub_domain, domain_name, domain_suffix, path

def url2vec_0 (input_string, input_domain):

  protocol, sub_domain, domain_name, domain_suffix, path = parse_url(input_string)
  new_url = protocol + '://' + input_domain + path
  try:
    if len(new_url) > 116:
      input_string = new_url[:116]
    special_value = len(label_encoder.classes_)  # Sử dụng một giá trị đặc biệt không trùng với các giá trị có sẵn trong label_encoder

    encoded_chars = [label_encoder.transform([char])[0] if char in label_encoder.classes_ else special_value for char in new_url]
    return encoded_chars
  except:
    a = 1

valid_characters = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "-", ".", "!", "*", "'", "(", ")", ";", ":", "&", "=", "+", "$", ",", "/", "?", "#", "[", "]", "@", "_", "%", " "
]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(valid_characters)

# Hàm này sẽ thay cái domain của url thành domain av đưa vào
def url2vec (input_string):

  new_url = input_string
  try:
    if len(new_url) > 116:
      input_string = new_url[:116]
    special_value = len(label_encoder.classes_)  # Sử dụng một giá trị đặc biệt không trùng với các giá trị có sẵn trong label_encoder

    encoded_chars = [label_encoder.transform([char])[0] if char in label_encoder.classes_ else special_value for char in new_url]
    return encoded_chars
  except:
    a = 1

#Data
d = 0
target_length = 116
# Đọc file CSV bằng Pandas
df = pd.read_csv("path_to_phishing sample")
df = df['url']
df = df.sample(n=5000, random_state=42)
df = df.tolist()

n = 3000 # number of attack sample
df_av = pd.read_csv("path_to_adv_sample") # mở file csv chứa mẫu phishing
df_av = df_av['adv']
df_av = df_av.tolist()

templ = []

for i in range(n):
  # if isinstance(i, str):
    temp = url2vec_0(df[i],df_av[i])
    try:
      if len(temp) < target_length:
            # Padding thêm 0 vào cuối mảng
            temp = temp + [0] * (target_length - len(temp))
      else:
            # Cắt bớt phần dư
            temp = temp[:target_length]
      templ.append(temp)
    except:
      d+=1

templ = np.array(templ) 
labels = np.ones(n)

from keras.models import load_model
from sklearn.metrics import accuracy_score
model = load_model("path_to_EMAN_modedl")

test_predict=model.predict(templ)
a = []
for i in test_predict:
  if(i>=0.5):
    a.append(1.0)
  else:
    a.append(0.0)

ac = accuracy_score(labels, a)
print('Detection rate adv: ', ac)