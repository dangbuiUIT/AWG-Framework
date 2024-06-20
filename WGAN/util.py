import tldextract
import random
import validators
import string
import numpy as np
import pandas as pd
from numpy import zeros
from numpy import ones

from numpy.random import randn
from numpy.random import randint
from collections import Counter

# define dictionary
def setup_dictionary():
    alphabet = string.ascii_lowercase + string.digits + "._-"
    dictionary_size = len(alphabet) + 1
    dictionary = {}
    reverse_dictionary = {}
        for i, c in enumerate(alphabet):
        dictionary[c]=i+1
        reverse_dictionary[i+1]=c
    return dictionary, reverse_dictionary, dictionary_size

dictionary, reverse_dictionary, dictionary_size = setup_dictionary()


def array2domain (domain):
  this_domain_gen = ""
  for position in domain:
    this_index = np.argmax(position)
    if this_index != 0:
        this_domain_gen += reverse_dictionary[this_index]
  return this_domain_gen

def domain2array (line):
  this_sample=np.zeros(domain_shape)
  p = False
  line = line.lower()
  if len ( set(line) - set(alphabet)) == 0 and len(line) <= domain_len and line != '':
    p = True
    for i, position in enumerate(this_sample):
        this_sample[i][0]=1.0

    for i, char in enumerate(line):
        this_sample[i][0]=0.0
        this_sample[i][dictionary[char]]=1.0

  return this_sample, p

def domain2image (domain):
  p = False
  if(len(domain) > domain_len):
    return None ,p
  arr = []
  for i in domain:
    if i not in alphabet:
      return [] ,p
    arr.append((ord(i)- 127.5) / 127.5)
  for i in range(domain_len-len(arr)):
    arr.append((0- 127.5) / 127.5)
  arr = np.array(arr)
  arr = (arr.reshape((6,6,1))).astype(np.float32)
  return arr, True

# generate dga domain based CHARBOT
def generate_dga_domain(d):

    extracted = tldextract.extract(d)
    domain_name = extracted.domain
    sub = extracted.subdomain
    tld = extracted.suffix
    # Step 3: Randomly select two indices i and j
    domain_name += tld
    i = random.randint(0, len(domain_name) - 1)

    # Step 4: Randomly select two replacement characters c1 and c2
    characters = "abcdefghijklmnopqrstuvwxyz0123456789-"
    c1 = random.choice(characters)
    while c1 == d[i]:
        c1 = random.choice(characters)

    # Convert d to a list to modify the characters
    domain_name = list(domain_name)

    # Step 5: Set d[i] = c1 and d[j] = c2
    domain_name[i] = c1
    domain_name.insert(len(domain_name)-len(tld),'.')

    if (sub != ''):
      domain_name = list(sub) + ['.'] + domain_name
    # Step 7: Return d:t
    return ''.join(domain_name)

def load_real_samples(path_data_set):
  df = pd.read_csv(path_data_set, header=None, names=['col1'])
  return df
# load image data
datasetr = load_real_samples()

def load_fake_sample(dataset_base):
  len = dataset_base.shape[0]
  dataset = []
  for i in range(len):
    item = datasetr['col1'].iloc[i]
    dataset.append(generate_dga_domain(item))
  return dataset

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = []
	for i in ix:
		item = dataset['col1'].iloc[i]
		temp, p = domain2array(item)
		temp = np.reshape(temp,(25,40,1))
		X.append(temp)
	# generate class labels, -1 for 'real'
	return X

def generate_fake_samples(dataset, n_samples):
	# choose random instances
	items = random.sample(dataset, n_samples)
	X = []
	for item in items:
		temp, p = domain2array(item)
		temp = np.reshape(temp,(25,40,1))
		X.append(temp)
	return X

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	return np.random.normal(0, 1, (n_samples, latent_dim))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, g_hist, path_save):
	# plot history
	pyplot.plot(d1_hist, label='crit')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig(path_save)
	pyplot.show()
	pyplot.close()