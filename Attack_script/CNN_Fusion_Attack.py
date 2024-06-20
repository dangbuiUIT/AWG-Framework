from urllib.parse import urlparse
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Input
import pandas as pd
import numpy as np
import idna

def parse_url(url):
    # Sử dụng hàm urlparse để phân tích URL thành các thành phần
    parsed_url = urlparse(url)

    # Lấy ra các trường thông tin từ URL
    protocol = parsed_url.scheme  # Protocol (http, https, ftp, etc.)
    sub_domain = parsed_url.hostname.split('.')[0]  # Sub-domain name
    domain_name = '.'.join(parsed_url.hostname.split('.')[1:-1])  # Domain name
    domain_suffix = parsed_url.hostname.split('.')[-1]  # Domain suffix (com, org, vn, etc.)
    path = ''  # URL path

    return protocol, sub_domain, domain_name, domain_suffix, path

def parse_url_0(url):
    # Sử dụng hàm urlparse để phân tích URL thành các thành phần
    parsed_url = urlparse(url)

    # Lấy ra các trường thông tin từ URL
    protocol = parsed_url.scheme  # Protocol (http, https, ftp, etc.)
    sub_domain = parsed_url.hostname.split('.')[0]  # Sub-domain name
    domain_name = '.'.join(parsed_url.hostname.split('.')[1:-1])  # Domain name
    domain_suffix = parsed_url.hostname.split('.')[-1]  # Domain suffix (com, org, vn, etc.)
    path = ''  # URL path

    res = protocol + '://' + sub_domain + '.'  + domain_name + '.' + domain_suffix
    return res



def convert_string_to_vector(input_string, l=150):
    char_to_index = {
        **{char: index + 1 for index, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
        **{char: index + 27 for index, char in enumerate("abcdefghijklmnopqrstuvwxyz")},
        **{char: index + 53 for index, char in enumerate("0123456789")},
        **{char: index + 63 for index, char in enumerate("-;.!?:&'\"/\\|_@#$%^&*~`+–=<>()[]{}")},
    }

    # Ký tự nằm ngoài đống trên sẽ được gán giá trị 96
    char_to_index.update({char: 96 for char in input_string if char not in char_to_index})

    vector = [char_to_index[char] for char in input_string]

    # Thêm padding nếu vector có chiều dài ngắn hơn l
    padding_value = 0
    while len(vector) < l:
        vector.append(padding_value)

    # Cắt đi các phần tử thừa nếu vector dài hơn l
    vector = vector[:l]

    return vector

def punycode_decode(domain):
  punycode_domain_bytes = idna.encode(domain)
  punycode_domain_str = punycode_domain_bytes.decode('utf-8')  # Chuyển đổi byte string thành chuỗi Unicode
  return punycode_domain_str

def process_url(url, av_domain):
  # try:
  #   av_domain = punycode_decode(av_domain)
  # except:
  #   av_domain = av_domain

  protocol, sub_domain, domain_name, domain_suffix, path = parse_url(url)
  new_url = protocol + '://' + av_domain
  url = convert_string_to_vector(new_url)
  return url

def process_url_0(url):
  url = parse_url_0(url)
  url = url.replace("..",".")
  url = convert_string_to_vector(url)
  return url

n = 1000 # number of attack sample
df_av = pd.read_csv("path_to_avd_sample") # mở file csv chứa mẫu phishing
av = df_av['adv']

import pandas as pd
# Đọc file CSV bằng Pandas
df = pd.read_csv("path_to_phishing_sample")
urls = df['url']
data = []
data1 = []
for i in range(n):
  temp =  process_url(urls[i%n], av[i])
  data.append(temp)
  temp = process_url_0(urls[i%n])
  data1.append(temp)

labels = np.ones(n)
data = np.array(data)
data1 = np.array(data1)

from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
model = load_model("path_to_CNN_Fusion")

test_predict=model.predict(data1)
a = []
for i in test_predict:
  if(i>=0.5):
    a.append(1.0)
  else:
    a.append(0.0)

ac = accuracy_score(labels, a)

print('Detection rate phish', ac)

test_predict=model.predict(data)
a = []
for i in test_predict:
  if(i>=0.5):
    a.append(1.0)
  else:
    a.append(0.0)

ac = accuracy_score(labels, a)

print('Detection rate avd', ac)