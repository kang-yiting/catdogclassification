# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:45:59 2019

@author: user
"""

#from PIL import Image確認圖片能導入
#img=Image.open('D:/kagglecatsanddogs_3367a/PetImages/Cat/500.jpg')

#匯入需要用到的模組、函式庫
import os #系統路徑相關的模組
import glob
import numpy as np #數列以及array相關的功能
from keras.preprocessing.image import  img_to_array, load_img #把圖檔轉換為array、讀取圖檔
from PIL import Image
#%%變數設定
dict_labels = {"Cat":0, "Dog":1} #labels的對應
size = (64,64) #由於原始資料影像大小不一，因此制定一個統一值
nbofdata=500   #從各個資料夾中抓取特定數量的檔案(這邊讀取500筆)
#%%
for folders in glob.glob("D:/kagglecatsanddogs_3367a/PetImages/*"): #執行次數=抓取幾個子資料夾
    print(folders) #第一次執行 變數值=Cat 第二次執行=Dog
    images=[] #清空記憶體 以免cat跟dog相加
    labels_hot=[]
    labels=[]
    nbofdata_i=1 #等等要用來控制讀取多少檔案用的
    
    for filename in os.listdir(folders): #抓取folders資料夾中的所有檔案
        if nbofdata_i <= nbofdata: #1~500
                    label = os.path.basename(folders)
                    className = np.asarray(label)
                    img=load_img(os.path.join(folders,filename))
                    img=img.resize(size,Image.BILINEAR)
                    if img is not None:
                        if label is not None:
                            labels.append(className)
                            labels_hot.append(dict_labels[label])
                        x=img_to_array(img)  #無法直接把image檔案疊加起來，必須要先把它變成array形式才能夠這麼操作
                        images.append(x)
                    nbofdata_i+=1
                    
#%%把疊加好之後的檔案儲存起來，另一個功能是檢查目前要儲存的路徑是否存在，如果沒有的話就建立一個
    images=np.array(images)    
    labels_hot=np.array(labels_hot)
    print("images.shape={}, labels_hot.shape=={}".format(images.shape, labels_hot.shape))    
    imagesavepath='Cat_Dog_Dataset/'
    if not os.path.exists(imagesavepath):
        os.makedirs(imagesavepath)
    np.save(imagesavepath+'{}_images.npy'.format(label),images)    
    np.save(imagesavepath+'{}_label.npy'.format(label),labels)    
    np.save(imagesavepath+'{}_labels_hot.npy'.format(label),labels_hot)
    print('{} files has been saved.'.format(label))