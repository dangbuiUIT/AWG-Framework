from urllib.parse import urlparse
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Input
import pandas as pd
import numpy as np
import random
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


def convert_string_to_vector(input_string, l=30):
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
    padding_value = 97
    while len(vector) < l:
        vector.append(padding_value)

    # Cắt đi các phần tử thừa nếu vector dài hơn l
    vector = vector[:l]

    return vector

    import numpy as np

def one_hot(input):
  # Số lượng categories
  num_categories = 98

  # Tạo ma trận one-hot encode
  one_hot_matrix = np.zeros((num_categories,len(input)))

  # Đánh dấu các vị trí tương ứng của số trong mỗi dòng
  for i, num in enumerate(input):
      one_hot_matrix[num, i] = 1  # Trừ 1 vì index trong Python bắt đầu từ 0

  # Reshape ma trận để có kích thước 97x30
  return one_hot_matrix

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
  av_domain = av_domain.split('.')
  if(len(av_domain) == 3):
    sub_domain = av_domain[0]
    domain_name = av_domain[1]
    domain_suffix = av_domain[2]
  else:
    domain_name = av_domain[0]
    domain_suffix = av_domain[1]
  protocol = convert_string_to_vector(protocol)
  protocol = one_hot(protocol)
  sub_domain = convert_string_to_vector(sub_domain)
  sub_domain = one_hot(sub_domain)
  domain_name = convert_string_to_vector(domain_name)
  domain_name = one_hot(domain_name)
  domain_suffix = convert_string_to_vector(domain_suffix)
  domain_suffix = one_hot(domain_suffix)
  path = convert_string_to_vector(path)
  path = one_hot(path)
  return protocol, sub_domain, domain_name, domain_suffix, path

def process_url_ori(url):
  protocol, sub_domain, domain_name, domain_suffix, path = parse_url(url)
  protocol = convert_string_to_vector(protocol)
  protocol = one_hot(protocol)
  sub_domain = convert_string_to_vector(sub_domain)
  sub_domain = one_hot(sub_domain)
  domain_name = convert_string_to_vector(domain_name)
  domain_name = one_hot(domain_name)
  domain_suffix = convert_string_to_vector(domain_suffix)
  domain_suffix = one_hot(domain_suffix)
  path = convert_string_to_vector(path)
  path = one_hot(path)
  return protocol, sub_domain, domain_name, domain_suffix, path

n = 5000 # Số mẫu attack đưa vào
df_av = pd.read_csv("/path_to_adv_sample") 
av = df_av['adv']

df = pd.read_csv("path_to_phishing") 
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []

data11 = []
data22 = []
data33 = []
data44 = []
data55 = []

for i in range(n):
  temp =  process_url(urls[i], av[i])
  data1.append(temp[0])
  data2.append(temp[1])
  data3.append(temp[2])
  data4.append(temp[3])
  data5.append(temp[4])

  temp =  process_url_ori(urls[i])
  data11.append(temp[0])
  data22.append(temp[1])
  data33.append(temp[2])
  data44.append(temp[3])
  data55.append(temp[4])

data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.array(data3)
data4 = np.array(data4)
data5 = np.array(data5)

data11 = np.array(data11)
data22 = np.array(data22)
data33 = np.array(data33)
data44 = np.array(data44)
data55 = np.array(data55)

labels = np.ones(n)
llabels = np.ones(nn)
from keras.models import load_model
from sklearn.metrics import accuracy_score
model = load_model("path_to_CCBLA_model")
test_predict=model.predict([data11,data22,data33,data44,data55])
a = []
for i in test_predict:
  if(i>=0.5):
    a.append(1.0)
  else:
    a.append(0.0)

ac = accuracy_score(labels, a)

print('Detection rate phish', ac)

test_predict=model.predict([data1,data2,data3,data4,data5])
a = []
for i in test_predict:
  if(i>=0.5):
    a.append(1.0)
  else:
    a.append(0.0)

ac = accuracy_score(labels, a)

print('Detection rate avd', ac)