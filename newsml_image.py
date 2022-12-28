# ライブラリ読み込み・初期化
# -*- coding: utf-8 -*-
import os
# import pwd
# import grp
import glob
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
# % matplotlib inline
import matplotlib.pyplot as plt

np.random.seed(5)
print("Init done.")

# リソースディレクトリの設定、配列追加
unclassified_label = "img"
output_path = "."
img_paths = []
for root, dirs, files in os.walk("./" + unclassified_label + "/"):
    for file in files:
        if file.endswith(".jpg"):
            img_paths.append(os.path.join(root, file))
img_num = len(img_paths)
print("Image number:", img_num)
print("Image list make done.")

# イメージサイズの取得
for i in img_paths[::124]:
    img = Image.open(i)
    print(img.size)
    plt.figure()
    plt.imshow(np.asarray(img))
print("Size Check done.")

# imgサイズの調整・配列に追加
plt.close("all")
img_paths = img_paths[:600]
print(len(img_paths))


def img_to_matrix(img):
    img_array = np.asarray(img)
    return img_array


def flatten_img(img_array):
    s = img_array.shape[0] * img_array.shape[1] * img_array.shape[2]
    img_width = img_array.reshape(1, s)
    return img_width[0]


dataset = []
for i in img_paths:
    img = Image.open(i)
    img = img.resize((int(1232 / 4), int(1754 / 4)), Image.BICUBIC)
    img = img_to_matrix(img)
    img = flatten_img(img)
    dataset.append(img)
dataset = np.array(dataset)
print(dataset.shape)
print("Dataset make done.")

# 次元数の圧縮
# batch_size < len(dataset) && n_components < batch_size である必要がある
n = dataset.shape[0]
batch_size = img_num
ipca = IncrementalPCA(n_components=batch_size - 1)
for i in range(n // batch_size):
    r_dataset = ipca.partial_fit(dataset[i * batch_size:(i + 1) * batch_size])
r_dataset = ipca.transform(dataset)
print(r_dataset.shape)
print("PCA done.")

# K-means によるクラスタリング
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=5).fit(r_dataset)
labels = kmeans.labels_
print("K-means clustering done.")
# uid = pwd.getpwnam("apache").pw_uid
# gid = grp.getgrnam("apache").gr_gid
for i in range(n_clusters):
    label = np.where(labels == i)[0]
    # Image placing
    if not os.path.exists(output_path + "/img_" + str(i)):
        os.makedirs(output_path + "/img_" + str(i))
    #       os.chown(output_path+"/img_"+str(i), uid, gid)
    for j in label:
        img = Image.open(img_paths[j])
        fname = img_paths[j].split('/')[-1]
        img.save(output_path + "/img_" + str(i) + "/" + fname)
#        os.chown(output_path+"/img_"+str(i)+"/" + fname, uid, gid)
print("Image placing done.")
