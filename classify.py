'''
  @author: Tiejian Zhang
  @email: zhangtj_pku@hotmail.com
  @method: SIFT -> k-means -> SVM
'''
'''
    原算法的问题是，针对每一个分类都制作了码本。然后在测试的时候是根据每个分类图片制作的特征向量的，其实这时应该假设不知道图片分类的。
'''
import cv2
import numpy as np

TrainSetInfo = {
	"car"		:	40,
	"city"		:	20,
	"dog"		:	30,
	"earth"		:	15,
	"fireworks"	:	20,
	"flowers"	:	20,
	"fruits"	:	20,
	"glass"		:	20,
	"gold"		:	15,
	"gun"		:	20,
	"plane"		:	40,
	"sky"		:	30,
	"worldcup"	:	40
}

TestSetInfo = {
	"car"		:	119,
	"city"		:	59,
	"dog"		:	49,
	"earth"		:	24,
	"fireworks"	:	54,
	"flowers"	:	63,
	"fruits"	:	78,
	"glass"		:	52,
	"gold"		:	44,
	"gun"		:	44,
	"plane"		:	102,
	"sky"		:	78,
	"worldcup"	:	131
}

def calcSiftFeature(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT(200) # max number of SIFT points is 200
	kp, des = sift.detectAndCompute(gray, None)
	return des
    
#求特征直方图
#这个函数可以使用以下语句替换
#from scipy.cluster.vq import *
#test_features = np.zeros((1, 50), "float32")
##descriptors是多个sift特征，voc是码本，vq返回每个sift特征对应的中心点索引，和距离中心点的距离
#words, distance = vq(descriptors,voc)
#for w in words:
#   test_features[0][w] += 1
#   return test_features
def calcFeatVec(features, centers):
	featVec = np.zeros((1, 50))
	for i in range(0, features.shape[0]):
		fi = features[i]
        #np.tile实现拼贴，变成50x50的矩阵
		diffMat = np.tile(fi, (50, 1)) - centers
        ##axis=0表示列求和，axis=1表示按行求和
		sqSum = (diffMat**2).sum(axis=1)    
		dist = sqSum**0.5
        #argsort表示升序排列时结果的索引
		sortedIndices = dist.argsort()
		idx = sortedIndices[0] # index of the nearest center
		featVec[0][idx] += 1
	return featVec

#逐个分类抽取sift特征值，保存文件
def initFeatureSet():
	for name, count in TrainSetInfo.items():
		dir = "TrainSet/" + name + "/"
        #创建0*128维度的数组，数据类型是float32
        #等同于 featureSet = np.zeros((0,128), dtype=np.float32)
		featureSet = np.float32([]).reshape(0,128)

		print "Extract features from training set" + name + "..."
		for i in range(1, count + 1):
			filename = dir + name + " (" + str(i) + ").jpg"
			img = cv2.imread(filename)
			des = calcSiftFeature(img)
			featureSet = np.append(featureSet, des, axis=0)
		
		featCnt = featureSet.shape[0]
		print str(featCnt) + " features in " + str(count) + " images\n"
		
		# save featureSet to file
		filename = "Temp/features/" + name + ".npy"
		np.save(filename, featureSet)
#针对每一个分类制作码本
def learnVocabulary():
	wordCnt = 50
	for name, count in TrainSetInfo.items():
		filename = "Temp/features/" + name + ".npy"
		features = np.load(filename)
		
		print "Learn vocabulary of " + name + "..."
		# use k-means to cluster a bag of features
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		compactness, labels, centers = cv2.kmeans(features, wordCnt, criteria, 20, flags)
		
		# save vocabulary(a tuple of (labels, centers)) to file
		filename = "Temp/vocabulary/" + name + ".npy"
		np.save(filename, (labels, centers))
		print "Done\n"

#逐个分类计算每个图片的码本直方图和所属分类结果
def trainClassifier():
	trainData = np.float32([]).reshape(0, 50)
	response = np.float32([])
	
	dictIdx = 0
	for name, count in TrainSetInfo.items():
		dir = "TestSet/" + name + "/"
		labels, centers = np.load("Temp/vocabulary/" + name + ".npy")
		
		print "Init training data of " + name + "..."
		for i in range(1, count + 1):
			filename = dir + name + " (" + str(i) + ").jpg"
			img = cv2.imread(filename)
			features = calcSiftFeature(img)
			featVec = calcFeatVec(features, centers)
			trainData = np.append(trainData, featVec, axis=0)
		
		res = np.repeat(np.float32([dictIdx]), count)
		response = np.append(response, res)
		dictIdx += 1
		print "Done\n"

	print "Now train svm classifier..."
	trainData = np.float32(trainData)
	response = response.reshape(-1, 1)  #变成n*1的二维数组
	svm = cv2.SVM()
    print trainData.shape, trainData.dtype, response.shape, response.dtype
	svm.train_auto(trainData, response, None, None, None) # select best params
	svm.save("svm.clf")
	print "Done\n"
	
def classify():
	svm = cv2.SVM()
	svm.load("svm.clf")
	
	total = 0; correct = 0; dictIdx = 0
	for name, count in TestSetInfo.items():
		crt = 0
		dir = "TestSet/" + name + "/"
		labels, centers = np.load("Temp/vocabulary/" + name + ".npy")
		
		print "Classify on TestSet " + name + ":"
		for i in range(1, count + 1):
			filename = dir + name + " (" + str(i) + ").jpg"
			img = cv2.imread(filename)
			features = calcSiftFeature(img)
			featVec = calcFeatVec(features, centers)
			case = np.float32(featVec)
			if (dictIdx == svm.predict(case)):
				crt += 1
			
		print "Accuracy: " + str(crt) + " / " + str(count) + "\n"
		total += count
		correct += crt
		dictIdx += 1
		
	print "Total accuracy: " + str(correct) + " / " + str(total)

if __name__ == "__main__":	
	#initFeatureSet()
	#learnVocabulary()
	trainClassifier()
	classify()