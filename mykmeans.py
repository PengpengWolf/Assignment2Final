# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:50:01 2020

@author: irisy
"""

   
import numpy as np
#import time
import matplotlib.pyplot as plt 
import random
 
# calculate Euclidean distance
def euclDistance(dataset, center):
    sum=0
    #for i in range(len(dataset)):
    #sum+=((dataset[i][0]-center[0])**2+(dataset[i][1]-center[1])**2)   
    sum+=((dataset[0]-center[0])**2+(dataset[1]-center[1])**2)     
    sum=np.sqrt(sum)
    return sum
def tolerance(data1, data2):
    data=[]
    sum=0
    if len(data1)==len(data2):
        for i in range(len(data1)):
            data[i]=((data1[i][0]-data2[i][0])**2+(data1[i][1]-data2[i][1])**2)
    sum=np.sum(data)
    #sum+=((dataset[i][0]-center[0])**2+(dataset[i][1]-center[1])**2)   
    return sum


# init centroids with random samples
def initCentroids(dataSet, k):
    
    numSamples, colum_1, colum_2 = np.shape(dataSet)
    centroids = np.zeros((k, colum_2))
    for i in range(k):
        index_x= random.randint(0, numSamples-1)
        index_y= random.randint(0, colum_1-1)
        #print(index_x)
        #print(index_y)
        centroids[i] = dataSet[index_x][index_y]
    return centroids

 
# k-means cluster
def kmeans(dataSet, k, maxiter):
    numSamples, colum_1, colum_2 = np.shape(dataSet)
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    #clusterAssment = np.mat(np.zeros((numSamples, 2)))
    clusterAssment = np.zeros((numSamples, colum_1,colum_2))
    clusterChanged = True
    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)
    #print(centroids)
    n=0

    while clusterChanged == True and n < maxiter:
        n+=1
        #print('try')
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist  = 100000.0
            minIndex = -1
            ## for each centroid
            ## step 2: find the closest centroid 
            for j in range(colum_1):
                for l in range(k):
                    distance = euclDistance(dataSet[i][j], centroids[l])
                    if distance < minDist:
                        minDist  = distance
                        minIndex = l
                clusterChanged = True
                clusterAssment[i][j]= [minIndex, minDist**2]
#            for j in range(k):
                #print(j)
#                distance = euclDistance(dataSet[i], centroids[j])
#                if distance < minDist:
#                    minDist  = distance
#                    minIndex = j
              ## step 3: update its cluster
#                clusterChanged = True
#                clusterAssment[i] = [minIndex, minDist**2]
        #print('k=',k)
        #print(clusterAssment)
        #update the centroids with mean values
        for p in range(k):
            #print('center')
            #print(p)
            sum0=0
            sum1=0
            pointInCluster=[]
            for q in range(numSamples):
                for s in range(colum_1):
                    if clusterAssment[q][s][0]==p:
                        pointInCluster.append(dataSet[q][s])
#                if clusterAssment[q][0]==p:
#                    pointInCluster.append(dataSet[q])
            #print(pointInCluster)
            if (len(pointInCluster)) >0 :
                for r in range(len(pointInCluster)):
                    sum0+=pointInCluster[r][0]
                    sum1+=pointInCluster[r][1]
                centroids[p][0]=sum0/(len(pointInCluster))
                centroids[p][1]=sum1/(len(pointInCluster))

            #print('len(pointInCluster',len(pointInCluster))

        #print('curr=',curr_cen)
       # print('pre=',pre_cen)
        #toler=tolerance(curr_cen,pre_cen)


    print('Congratulations, cluster complete!')
    #print('n==',n)
    return centroids, clusterAssment
 
# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, colum_1, colum_2 = np.shape(dataSet)
    dataset_tags=[]
    mark = ['or', 'ob', 'og', 'oc', 'oy']

 
    # draw all samples
    for i in range(numSamples):
        dataset_tag=[]
        for j in range(colum_1):
            #print(i)
            #print(j)
            markIndex = int(clusterAssment[i][j][0])
#            plt.plot(dataSet[i][j][0], dataSet[i][j][1],mark[markIndex])
#        markIndex = int(clusterAssment[i][0])
       # print(markIndex)
            dataset_tag.append((dataSet[i][j],markIndex))
        #print(dataset_tag)
        dataset_tags.append(dataset_tag)
    mark = ['ok', 'pk', 'sk', '*k', '+k']
    # draw the centroids
#    for i in range(k):
#        plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 20)
 
    #plt.show()
    return dataset_tags


def countTag(dataset_tags,k):
    num=np.zeros(k)
    numSamples, colum_1, colum_2 = np.shape(dataset_tags)
    for i in range(numSamples):
        for j in range(colum_1):
            if dataset_tags[i][j][1]==0:
                num[0]+=1
            if dataset_tags[i][j][1]==1:
                num[1]+=1
            if dataset_tags[i][j][1]==2:
                num[2]+=1
            if dataset_tags[i][j][1]==3:
                num[3]+=1
            if dataset_tags[i][j][1]==4:
                num[4]+=1
    return num