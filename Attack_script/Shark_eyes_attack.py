from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import idna

characters = "abcdefghijklmnopqrstuvwxyz0123456789-. "
characters= list(characters)
characters = array(characters)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(characters)

max_len_domain = 100

with open('path_to_file_tag_list.txt', 'r') as file:
    content = file.read()
tags = ast.literal_eval(content)
tags = list(set(tags))
tags = array(tags)
label_encoder2 = LabelEncoder()
integer_encoded = label_encoder2.fit_transform(tags)

def punycode_decode(domain):
  punycode_domain_bytes = idna.encode(domain)
  punycode_domain_str = punycode_domain_bytes.decode('utf-8')  # Chuyển đổi byte string thành chuỗi Unicode
  return punycode_domain_str

max_len_content = 2500

def domain2vec (input_string):
  try:
    input_string = punycode_decode(input_string)
  except:
    input_string = input_string
  encoded_chars = []
  for char in input_string:
    try:
      encoded_chars.append(label_encoder2.transform([char])[0]+1)
    except:
      encoded_chars.append(40)

  if len(encoded_chars) > 100:
    encoded_chars = encoded_chars[:100]
  else:
    while len(encoded_chars) < 100:
        encoded_chars.append(41)
  return encoded_chars

def html2vec (input_list):
  encoded_chars = []
  if (len(input_list) > 2500):
    input_list = input_list[:2500]
  for char in input_list:
    try:
      encoded_chars.append(label_encoder2.transform([char])[0]+1)
    except:
      encoded_chars.append(2706)

  if len(encoded_chars) < 2500:
    encoded_chars = encoded_chars + [2707] * (2500 - len(encoded_chars))

  return encoded_chars

def domain2image (input_string):
  try:
    input_string = punycode_decode(input_string)
  except:
    input_string = input_string
  int_list = []
  for i in input_string:
    int_list.append(ord(i))

  if len(int_list) > 100:
    int_list = int_list[:100]
  else:
    while len(int_list) < 100:
        int_list.append(0)
  return int_list

av_domain =  []
n = 1500 # number of attact sampple
df_av = pd.read_csv("path_to_adv_sample") # mở file csv chứa mẫu phishing
av = df_av['adv_homo']

df = pd.read_csv("path_to_phishing_sample")
domls = df['doml']

imgl = []
strucl = []
strucdom = []
for i in range(n):
  t1 = np.array(domain2vec(av[i]))
  strucl.append(t1)
  t2 = np.array(domain2image(av[i])).reshape((10,10,1))
  imgl.append(t2)
  t3 = np.array(html2vec(eval(domls[i])))
  strucdom.append(t3)

from keras.models import load_model
model_load = load_model("path_to_sharkeyes_model")
test_predict=model_load.predict([imgl,strucl,strucdom])
a = []
for i in test_predict:
  if(i>=0.5):
    a.append(1.0)
  else:
    a.append(0.0)

ac = accuracy_score(labels, a)

print("Detection rate adv: ", ac)