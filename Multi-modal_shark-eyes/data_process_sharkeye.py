from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

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

def domain2vec (input_string):
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
  for char in input_list:
    try:
      encoded_chars.append(label_encoder2.transform([char])[0]+1)
    except:
      encoded_chars.append(2706)

  if len(encoded_chars) > 2500:
    encoded_chars = encoded_chars[:2500]
  else:
    encoded_chars = encoded_chars + [2707] * (2500 - len(encoded_chars))

  return encoded_chars

def domain2image (input_string):
  int_list = []
  for i in input_string:
    int_list.append(ord(i))

  if len(int_list) > 100:
    int_list = int_list[:100]
  else:
    while len(int_list) < 100:
        int_list.append(0)
  return int_list

# visual
def visualing(path_dataset, path_visual):
  df = pd.read_csv("path_dataset") # load cac tập dataset
  domains = df['domain'] # thay bang av_domain
  domls = df['doml']

  imgl = []
  strucl = []
  strucdom = []
  for i in range(len(domains)):
    t1 = domain2vec(domains[i])
    strucl.append(t1)
    t2 = domain2image(domains[i])
    imgl.append(t2)
    t3 = html2vec(eval(domls[i]))
    strucdom.append(t3)

  df = pd.DataFrame({"1": imgl, "2": strucl, "3": strucdom})
  df.to_csv("path_visual", index=False)

