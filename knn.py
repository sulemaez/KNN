import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random

# KNN classifier class
class KNN:

    #set the training dataset
    def train(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    #get the prediction
    def predict(self,x_test,k = 1):
        self.k = k
        predictions = []   
        #for each test find prediction
        for row in x_test:
            label = self.get_prediction(row)
            predictions.append(label)
        return predictions    
    
    #get the predictions
    def get_prediction(self,row):
        found = []
        labels = []
        
        #get the k nearest neighbours
        for i in range(0,self.k):
            label , index = self.get_least(row,found)
            found.append(index)
            labels.append(label)
         #get the most occuring label
        return max(set(labels), key=labels.count)

    #gets the closest neighbour from the remaining 
    def get_least(self,row,found):
         #set the beginning
        best_dist = math.inf
        best_index = 0
        #loop through all training set getting closest point 
        for i in range(1,len(self.x_train)):
            dist =  self.euc(row,self.x_train[i])
            #if this is closer set and not found earlier 
            if dist < best_dist and i not in found:
                best_dist = dist
                best_index = i 
        #return the y_value        
        return self.y_train[best_index] , best_index


    #get the eucledian distance 
    def euc(self,a,b):
        distance = 0
        for x in range(len(a)):
            distance = pow((a[x] -b[x]), 2)
        
        return math.sqrt(distance)   

#finds the accuracy of aour algorithm
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

if __name__ == "__main__":
    
    knn = KNN()

    #load dataset and divide it 
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = .3)  

    #train model 
    knn.train(x_train,y_train)
    
    #get the predictions 
    predictions = knn.predict(x_test,10) 

    acc = getAccuracy(y_test,predictions)

    print(" {}% accurate ".format(acc))



  





