import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
import sys
import numbers
import argparse
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import time
from Dictionaries import countryCodes
from Dictionaries import currencyCodes
from Dictionaries import kickstarterCategories
from sklearn.preprocessing import StandardScaler

################################prepare data for use in scikit-learn###############################################

kData = pandas.read_csv("Kickstarter_Altered_Tester.csv", dtype = {"goal": float, "state": int, "disable_communication": int, "country": object, "currency": object, "staff_pick": int, "category": object,})
kDataInput = pandas.read_csv("Kickstarter_Altered_Tester_Blank.csv", dtype = {"goal": float, "disable_communication": int, "country": object, "currency": object, "staff_pick": int, "category": object,})

#collapse all textual columns to numerical

goalList = kData['goal'].tolist()
stateList = kData['state'].tolist()
disComList = kData['disable_communication'].tolist()
countryList = kData['country'].tolist()
for index, val in enumerate(countryList):
    countryList[index] = countryCodes[val]
currencyList = kData['currency'].tolist()
for index, val in enumerate(currencyList):
    currencyList[index] = currencyCodes[val]
staffPickList = kData['staff_pick'].tolist()
categoryList = kData['category'].tolist()
for index, val in enumerate(categoryList):
    categoryList[index] = kickstarterCategories[val]

kData['goalN'] = kData['goal'];
kData['stateN'] = kData['state'];
kData['disable_communicationN'] = kData['disable_communication'];
kData['countryN']=pandas.Series(countryList)
kData['currencyN']=pandas.Series(currencyList)
kData['staff_pickN'] = kData['staff_pick'];
kData['categoryN']=pandas.Series(categoryList)

#drop original text columns

kData = kData.drop(['goal','state','disable_communication','country','currency','staff_pick','category'],1)

#shuffle kData

kData = kData.sample(frac=1)

########################################################################## scikit-learn ################################################################

#train and test splits

def GetTrainTest(kData, yColumn, ratio):
    randomMask = np.random.rand(len(kData)) < ratio
    kData_train = kData[randomMask]
    kData_test = kData[~randomMask]
    
    Y_train = kData_train[yColumn].values
    Y_test = kData_test[yColumn].values
    del kData_train[yColumn]
    del kData_test[yColumn]
 
    X_train = kData_train.values
    X_test = kData_test.values
    return X_train, Y_train, X_test, Y_test
	
yColumn = 'stateN'
trainTestRatio = 0.7
X_train, Y_train, X_test, Y_test = GetTrainTest(kData, yColumn, trainTestRatio)

#the data is now ready for machine learning

dictClassifiers = {
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators = 18),
    "Decision Tree": tree.DecisionTreeClassifier(),
	"Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Neural Net": MLPClassifier(alpha = 1),
}

numberOfClassifiers = len(dictClassifiers.keys())

#Prediction method. Also re-splits and retrains at every iteration if slow mode is selected
 
def performClassification(fast, goal, disableCommunication, country, currency, staffPick, category, X_train, Y_train, X_test, Y_test):
	kDataResults = pandas.DataFrame(data=np.zeros(shape=(numberOfClassifiers,4)), columns = ['classifier', 'trainScore', 'testScore', 'timeToTrain'])
	testScoreList = list()
	count = 0
	finalCount = 0
	testScoreTotal = 0
	
	kDataInput['goal'] = pandas.Series(goal)
	kDataInput['disable_communication'] = pandas.Series(disableCommunication)
	kDataInput['country'] = pandas.Series(countryCodes[country])
	kDataInput['currency'] = pandas.Series(currencyCodes[currency])
	kDataInput['staff_pick'] = pandas.Series(staffPick)
	kDataInput['category'] = pandas.Series(kickstarterCategories[category])
	
	for key, classifier in dictClassifiers.items():	
		
		timeCount = 0
		t_start = time.clock()
		classifier.fit(X_train, Y_train)
		scaler = StandardScaler().fit(X_train)
		t_end = time.clock()
		timeChange = t_end - t_start
		trainScore = classifier.score(X_train, Y_train)
		testScore = classifier.score(X_test, Y_test)
		kDataResults.loc[count,'classifier'] = key
		kDataResults.loc[count,'trainScore'] = trainScore
		kDataResults.loc[count,'testScore'] = testScore
		kDataResults.loc[count,'timeToTrain'] = timeChange	
		testScoreList.append(testScore)
		if(fast == 1):
			rangeLimit = 5
		else:
			rangeLimit = 6
		successCount = 0
		timeCount = timeCount + timeChange
		
		for x in range (0,rangeLimit):
			if(fast == 0):
				X_train, Y_train, X_test, Y_test = GetTrainTest(kData, yColumn, trainTestRatio)		
				classifier.fit(X_train, Y_train)
			scaler = StandardScaler().fit(X_train)
			t_end = time.clock()
			timeChange = t_end - t_start
			trainScore = classifier.score(X_train, Y_train)
			testScore = classifier.score(X_test, Y_test)
			kDataResults.loc[count,'classifier'] = key
			kDataResults.loc[count,'trainScore'] = trainScore
			kDataResults.loc[count,'testScore'] = testScore
			kDataResults.loc[count,'timeToTrain'] = timeChange		
			if(classifier.predict(kDataInput) == 1):
				successCount+=1
				
		timeCount = timeCount + timeChange				

		testNumber = count	
		print(key, end = ', ')
		print(str(round((successCount/rangeLimit)*100, 2)), end = ", ")
		print(str(round((testScore*100),2)))
		
		count+=1
	
	return kDataResults
	
inputError = 0
fastError = 0
goalError = 0
disComError = 0
countryError = 0
currencyError = 0
staffPickError = 0
categoryError = 0	
	

try:	
	fastArg = int(sys.argv[1])
except ValueError:
	fastArg = -1	
	fastError = 1
try:		
	goalArg = float(sys.argv[2])
except ValueError:
	goalArg = -1
	goalError = 1
try:
	disComArg = int(sys.argv[3])
except ValueError:
	disComArg = -1
	disComError = 1
countryArg = sys.argv[4]
currencyArg = sys.argv[5]
try:
	staffPickArg = int(sys.argv[6])
except ValueError:
	staffPickArg = -1
	staffPickError = 1
categoryArg = sys.argv[7]


if(fastArg != 0 and fastArg != 1):
	inputError = 1
	fastError = 1
if(isinstance(goalArg, numbers.Real) == 0 or goalArg < 0):
	inputError = 1
	goalError = 1
if(disComArg != 0 and disComArg != 1):
	inputError = 1
	disComError = 1
	print(disComArg)
if((countryArg in countryCodes) == 0):
	inputError = 1
	countryError = 1
if((currencyArg in currencyCodes) == 0):
	inputError = 1
	currencyError = 1
if(staffPickArg != 0 and staffPickArg != 1):
	inputError = 1
	staffPickError = 1
	print(staffPickArg)
if((categoryArg in kickstarterCategories) == 0):
	inputError = 1
	categoryError = 1
	
if (inputError == 1):
	print("There was an issue with the entered field(s)", end = '')
	if(fastError ==1):
		print("; forecast", end = '')
	if(goalError ==1):
		print("; original pledge goal", end = '')		
	if(disComError ==1):
		print("; disabled communication", end = '')
	if(countryError ==1):
		print("; country", end = '')
	if(currencyError ==1):
		print("; currency", end = '')
	if(staffPickError ==1):
		print("; staff pick", end = '')
	if(categoryError ==1):
		print("; project category", end = '')	
else:
	kDataResults = performClassification(fastArg, goalArg, disComArg, countryArg, currencyArg, staffPickArg, categoryArg, X_train, Y_train, X_test, Y_test)

sys.stdout.flush()
