
sift + kmeans  + td-idf + svm实现图片分类

1、使用opencv提取sift特征

2、使用scipy的kmeans api计算聚类(速度要不cv2.kmeans算法快很多)

3、训练svm分类器

4、测试预测准确率


调参测试码本数量对分类预测准确率的影响
码本数量   抽取sift特征数   是否引入td-idf    合计抽取的特征数       准确率          二次测试准确率
2000        200           noidf           61837         0.66889632107
2000            max           noidf           157434        0.758082497213      0.76254180602
2000            max           idf             157434        0.78149386845       0.779264214047

50              max           noidf           157434        0.643255295429      0.59977703456
1000            max           idf             157434        0.746934225195
4000            max           idf             157434        0.748049052397      0.761426978818
10000           max           idf             157434        0.693422519509   


1、码本数量2000时效果最好，准确率超过80%
2、抽取sift特征越多，效果越好
3、加入tf-idf效果会有提升 

