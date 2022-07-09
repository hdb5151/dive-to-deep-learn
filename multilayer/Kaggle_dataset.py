import hashlib
import os
import tarfile
import zipfile
import requests

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')): #@save  print(os.path.join('a','b','c')) == a/b/c
      """下载⼀个DATA_HUB中的⽂件，返回本地⽂件名"""
      assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
      url, sha1_hash = DATA_HUB[name]
      os.makedirs(cache_dir, exist_ok=True)
      fname = os.path.join(cache_dir, url.split('/')[-1])
      if os.path.exists(fname):
            sha1 = hashlib.sha1()
            with open(fname, 'rb') as f:
                  while True:
                        data = f.read(1048576)
                        if not data:
                              break
                        sha1.update(data)
            if sha1.hexdigest() == sha1_hash:
                  return fname # 命中缓存
      print(f'正在从{url}下载{fname}...')
      r = requests.get(url, stream=True, verify=True)
      with open(fname, 'wb') as f:
            f.write(r.content)
      return fname

def download_extract(name, folder=None): #@save
      """下载并解压zip/tar⽂件"""
      fname = download(name)
      base_dir = os.path.dirname(fname)
      data_dir, ext = os.path.splitext(fname)
      if ext == '.zip':
            fp = zipfile.ZipFile(fname, 'r')
      elif ext in ('.tar', '.gz'):
            fp = tarfile.open(fname, 'r')
      else:
            assert False, '只有zip/tar⽂件可以被解压缩'
      fp.extractall(base_dir)
      return os.path.join(base_dir, folder) if folder else data_dir

def download_all(): #@save
      """下载DATA_HUB中的所有⽂件"""
      for name in DATA_HUB:
            download(name)



DATA_HUB['kaggle_house_train'] = ( #@save
      DATA_URL + 'kaggle_house_pred_train.csv',
      '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = ( #@save
      DATA_URL + 'kaggle_house_pred_test.csv',
      'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))


# print(train_data.shape)
# print(test_data.shape)

# 看看前四个和最后两个特征，以及相应标签（房价）。
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features)
