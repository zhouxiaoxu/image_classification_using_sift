# coding: utf-8
'''
    抽取图像的sift特征
    -》聚类生成码本
    -》得到训练集特征向量
    -》训练svm分类器
    -》测试分类器准确率
'''
import sys
sys.path.insert(0,'/home/zxy/opencv/opencv-2.4.13/build/lib')
import numpy as np
import cv2
import os
from scipy.cluster.vq import *
from sklearn import preprocessing

voc_cnt = 1000
trainset_path = 'TrainSet'
testset_path = 'TestSet'
'''
抽取图片的sift特征
'''
def calcSiftFeature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kps, des= sift.detectAndCompute(gray, None)
    return des

'''
计算图片的特征向量
'''
def calcImageFeature(des, centers):
    feature = np.zeros((1,voc_cnt), dtype=np.float32)
    words,distance = vq(des, centers)
    for i in words:
        feature[0][i] += 1
    return feature

'''
    计算所有训练图片的特征向量，保存在npy文件中
'''
def calcFeatureSet():
    dirs = os.listdir(trainset_path)
    deses = np.zeros((0,128), dtype=np.float32)
    img_cnt = 0
    for dir in dirs:
        print 'extract', dir, 'sift feature'
        files = os.listdir(os.path.join(trainset_path, dir))
        for f in files:
            img_cnt += 1
            im = cv2.imread(os.path.join(trainset_path, dir, f))
            des = calcSiftFeature(im)
            if des is not None:
                #注意如果没有axis =0，结果竖flatten成一个一维数组
                deses = np.append(deses, des, axis=0)
    print   img_cnt, 'images extract', deses.shape[0], 'sift features'
    np.save('Temp/train_sift_features.npy', deses)
    
def calcVoc():
    deses = np.load('Temp/train_sift_features.npy')
    #设置终止条件为误差值小于0.1或者迭代20次后
    criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 20, 0.1)
    print 'begin kmeans cluster'
    #_, labels, centers = cv2.kmeans(deses, voc_cnt,criteria,20,cv2.KMEANS_RANDOM_CENTERS)
    centers, variance = kmeans(deses, voc_cnt, 1) 
    print 'kmeans cluster done'
    np.save('Temp/voc.npy', centers)
    
idf = None
def trainClassfier():
    dirs = os.listdir(trainset_path)
    print 'trainClassfier', dirs
    centers = np.load('Temp/voc.npy')
    features = np.zeros((0,voc_cnt), dtype=np.float32)
    labels = np.float32([])
    dictIdx = 0
    
    print 'begin train classfier'
    for dir in dirs:
        files = os.listdir(os.path.join(trainset_path, dir))
        
        for f in files:
            im = cv2.imread(os.path.join(trainset_path, dir, f))
            des = calcSiftFeature(im)
            feature = calcImageFeature(des, centers)
            #也可以用这种表达方式features = np.vstack((features, features))
            features = np.append(features, feature, axis=0)
            
            #np.float32做类型转换，否则变成np.float64
            labels = np.append(labels, np.float32(dictIdx))
        dictIdx += 1
    #计算单词在所有特征出现的次数
    nbr_occurences = np.sum((features>0)*1, axis=0)
    #就算每个词的idf = log(文章总数/某个单词在多少篇文章中出现过)
    global idf
    idf = np.array(np.log((1.0*features.shape[0]+1)/(1.0*nbr_occurences+1)), dtype=np.float32)
    print features.shape, idf.shape
    features = features*idf
    features = preprocessing.normalize(features, norm='l2')
    
    labels = labels.reshape((-1,1))
    svm = cv2.SVM()
    print features.shape, features.dtype, labels.shape, labels.dtype
    svm.train_auto(features, labels, None, None, None)
    svm.save("svmV2.clf")
    print 'train classfier Done!'
    
def classify():
    svm = cv2.SVM()
    svm.load("svmV2.clf")
    centers = np.load('Temp/voc.npy')
    
    total = 0; correct = 0; dictIdx = 0
    dirs = os.listdir(testset_path)
    print 'start testing'
    print 'classify', dirs
    for dir in dirs:
        count = 0; crt = 0
        files = os.listdir(os.path.join(testset_path, dir))
        for f in files:
            count += 1
            im = cv2.imread(os.path.join(testset_path,dir,f))
            des = calcSiftFeature(im)
            feature = calcImageFeature(des, centers)
            feature = feature*idf
            feature = preprocessing.normalize(feature, norm='l2')
            if dictIdx == svm.predict(feature):
                crt += 1
        print 'Accuracy Class', dir, crt, '/', count, '=',float(crt)/count
        total += count
        correct += crt
        dictIdx += 1
    print 'Total Accuracy ', correct, '/', total, float(correct)/total
    

if __name__ ==  '__main__':
    calcFeatureSet()
    calcVoc()
    trainClassfier()
    classify()
            
            
            
            
            
