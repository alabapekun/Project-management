#!/usr/bin/env python
# coding: utf-8

# #Olanipekun Taofeeq Alaba
# #Data Mining Project
# #Online Payments Fraud Detection Dataset
# #Instructor: Professor Pablo Guillen Rondon

# # About the dataset and Project
# https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
#     
#     1.Introduction
# 
# In this project we are going to demonstrate the use of different machine learning algorithm both the traditional and auto machine learning to make classification.Using the computer language, supervised and non-supervised machine will be used. We will build models, fit models and make prediction with our models.The result from the adopted machine learning algorithms will then be compared. This is to determine the best machine learning algorithm for our dataset. However, before data is passed to our machine, the dataset will be explored, visualisation will be provided for better understanding. This project sourced it dataset from kaggle, online payment fraud detection dataset was adopted for this project. The link to the dataset is hereby provided https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset. The dataset consist of eleven columns. The first column provides data on step. Step represents a unit of time where 1 step equals 1 hour. The second column supply information on type of payment. There are 5 different type of payment namely:payment, transfer, cash out, debit and cash in. The third column had the amount. This is the amount involved in a particular transaction. The fourth column is "nameOrig". it implies that customer starting the transaction. Column fifth provide information on oldbalanceOrg. This is the balance before the transaction was made by the customer. Sixth column deals with 'newbalanceOrig': This is the balance after the customer transaction either successful or not. nameDest is the seventh column. It provides information on recipient of the transaction.
# The eighth column is 'oldbalanceDest'. This is the initial balance of recipient before the transaction was made. The next column is the 'newbalanceDest'.This is the new balance of recipient after the transaction was made. The sceond to the last column deals with isFraud. This categorized a particular transaction as either fraud or not. The last column is 'is flaggedfraud. The online payments fraud detection dataset consist of 11 features and 1048575 samples. Each row of the table represent an online payment transaction. 

# ##Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.utils import resample 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


pwd


# #import, read and exploration of online payment fraud detection dataset.
# Getting the data
# we can get the datasets using pandas library. Then we explore the dataset. looking out for null value and pay attention to categorical variable. we investigate if there is inbalance in the dataset. Find out if all columns are relevant otherwise drop columns that does not impact the dataset.
# 

# In[3]:


df = pd.read_csv("C:\\Users\\Alaba\\Desktop\\Data Mining\\Payments Fraud Detection Dataset.csv")
df.head()


# Above displayed the first 5 rows of all the dataset

# In[4]:


df.tail()


# Above displayed the last 5 rows of all the dataset

# In[5]:


df.columns


# Above display the feature name in the dataset

# In[6]:


df.isna().sum()


# In[7]:


df.isnull().sum()


# Here two different code were run to be double sure there is no null value in the dataset

# In[8]:


df[df['isFraud'].isna()]


# In[9]:


df[df['isFlaggedFraud'].isna()]


# In[10]:


df[df['amount'].isna()]


# The above investigate particular features to ensure there are no null values.

# In[11]:


df.describe()


# In[12]:


df.dtypes


# With the result above the dataset consist of integer, float and object type of dataset

# Let take a look at the histogram of all the variables in the dataset 

# In[13]:


df.hist(bins =50, figsize = (20,10))


# In[14]:


df['isFraud'].hist()


# #The histogram above gives us an indepth understanding of 'isFraud', from the histogram is obviuos that there is a case of inbalance problem, the zero are more than the 1

# #Now we are plotting the graphs by comparing each of the columns. First we compare the new balance origin to new balance destination and we compare old balance origin to old balance destination. 

# In[15]:


sns.FacetGrid(df,hue='isFraud',height = 10).map(plt.scatter,'newbalanceOrig','newbalanceDest').add_legend()


# In[16]:


sns.FacetGrid(df,hue='isFraud',height = 10).map(plt.scatter,'oldbalanceOrg','oldbalanceDest').add_legend()


# Below is the result of counts, '0' has 1047433 while '1' has 1142. This is a clear inbalance problem

# In[17]:


df['isFraud'].value_counts()


# In[18]:


df[['step','isFraud']].plot(kind = 'scatter', x = 'step', y= 'isFraud',figsize= (20,10))


# Above is the scatter of 'isFraud' note the presence of outlier

# In[19]:


df[['amount','isFraud']].plot(kind = 'scatter', x = 'amount', y= 'isFraud',figsize= (20,10))


# Let explore the feature isFlaggedFraud, it was discovered that it consist of only '0'. This variable was later dropped has it does not impact the dataset in anywhere.

# In[20]:


df['isFlaggedFraud'].value_counts()


# In[21]:


df['isFlaggedFraud'].hist()


# In[22]:


df.columns


# In[23]:


print(df['isFlaggedFraud'].value_counts())
print(len(df['isFlaggedFraud']))


# #drop isflaggedfraud, the value is constant it does not impact the dataset

# In[24]:


dfnew = df[['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
     'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']]
dfnew


# In[25]:


dfnew.head() 


# In[26]:


dfnew.tail()


# #Rename column isFraud to FRAUD

# In[27]:


dfnew.rename({'isFraud':'FRAUD'}, axis = 'columns', inplace =True)


# In[28]:


dfnew.head()


# In[29]:


dfnew['FRAUD'].unique()


# In[30]:


dfnew['type'].unique()


# In[31]:


dfnew['step'].unique()


# In[32]:


dfnew['nameOrig'].value_counts() 


# In[33]:


dfnew['nameDest'].value_counts() 


# #We do not need the name origination and name destination, the information was there to let us know where the fund is coming from and where is going to. As such we are going to drop the two columns

# In[34]:


dfnew.drop(columns= ['nameOrig','nameDest'], axis = 1, inplace = True)


# In[35]:


dfnew


# In[36]:


dfnew.dtypes


# #To balance the inbalance using the up sampling approach

# In[37]:


from sklearn. utils import resample
isNotFraud = dfnew[dfnew['FRAUD']== 0]
Fraud = dfnew[dfnew['FRAUD']== 1]


# In[38]:


isNotFraud.head()


# In[39]:


len(isNotFraud)


# In[40]:


Fraud.head()


# In[41]:


len(Fraud)


# In[116]:


from sklearn. utils import resample
unsampled = resample (isNotFraud,
                    replace =True,
                    n_samples = len(Fraud),
                    random_state =28)


# In[43]:


unsampled = pd.concat([Fraud,unsampled])
unsampled


# #check new class of count

# In[44]:


unsampled['FRAUD'].value_counts()


# In[45]:


unsampled['FRAUD'].hist()


# The problem of inbalance is solved

# #find out if there is a problem of multiple linearity

# In[46]:


plt.figure(figsize= (10,8))
sns.heatmap(unsampled.corr())


# #the step,amount and new balance destination has corelation with Fraud

# In[47]:


dfu =unsampled
cashout=dfu[dfu['type']=='CASH_OUT']
payment=dfu[dfu['type']=='PAYMENT']
cahin=dfu[dfu['type']=='CASH_IN']
transfer=dfu[dfu['type']=='TRANSFER']
debit=dfu[dfu['type']=='DEBIT']


# In[48]:


cashout


# #There were 990 cash out type of payment out of 1048575 online payment transaction

# In[49]:


payment


# In[50]:


Online_payment_counts = df['type'].value_counts().sort_index()
#use facies labels to index each count
type_labels = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
Classes_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00','#1B4F72']
Online_payment_counts.index = type_labels
Online_payment_counts.plot(kind='bar',color=Classes_colors,
                         title='Distribution of df by type')


# In[51]:


#Above provide us with 389 payments types of payment out of 1048575 online payment transaction


# In[52]:


#Converting String to continue values
dfu['PAYMENT']= np.where (dfu['type'] == 'PAYMENT',1,0)
dfu['TRANSFER']= np.where (dfu['type'] == 'TRANSFER',1,0)
dfu['CASH_OUT']= np.where (dfu['type'] == 'CASH_OUT',1,0)
dfu['DEBIT']= np.where (dfu['type'] == 'DEBIT',1,0)
dfu['CASH_IN']= np.where (dfu['type'] == 'CASH_IN',1,0)


# In[53]:


#Drop the original column 'type' from the dataframe
dfu.drop(columns=['type'],axis =1, inplace = True)


# In[54]:


dfu


# In[55]:


##Define x and y variables, then split the dataset into train and test
from sklearn.model_selection import train_test_split
x = dfu.drop('FRAUD',axis = 1)
y = dfu['FRAUD']


# In[56]:


x


# In[57]:


y


# In[58]:


dfu['FRAUD'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.show()


# We know that Fraud is categorized into two isNotFraud and Fraud. For every classification problems it is better if the datasets are balanced. We can say that the datasets is now balanced. 

# In[59]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state = 123)
x_train.shape


# In[60]:


##Standardization 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(x)
StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# ##  1. KNN

# In[61]:


#Fitting, training and predicting with the KNN model 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 51)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
y_pred


# In[62]:


#Training the KNN model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 51)
classifier.fit(x_train,y_train)


# In[63]:


#Accuracy
from sklearn.metrics import accuracy_score
modelacc = accuracy_score(y_test,y_pred)
modelacc


# In[64]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
conM = confusion_matrix(y_test,y_pred)
conM
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# The following were the correct classifications made by knn model 311 and 272. wrong classification were 27 and 76. in total we have 583 correct classifications and 103 wrong classifications. the algorithm had 85%. This was before optimization, let see what happened when we optimize.

# #Optimization for the value of K

# In[65]:


Error = []
# calculating errors for k value between 1 and 51
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    Error.append(np.mean(pred_i != y_test))


# In[66]:


plt.figure(figsize = (10, 6))
plt.plot(range(1,51), Error,color = "red", linestyle = 'dashed',marker = 'o',markerfacecolor = 'blue',markersize = 10)
plt.title('Error Rate for K value')
plt.xlabel('K value')
plt.ylabel('Mean Error')


# In[67]:


from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
k = np.random.randint(1,11,20)
param = {"n_neighbors" : k}
random_search = RandomizedSearchCV(classifier,param,n_iter = 5,cv = 5,n_jobs = -1,verbose = 0)
random_search.fit(x_train,y_train)


# In[68]:


print("train score-" + str(random_search.score(x_train,y_train)))
print("test score-" + str(random_search.score(x_test,y_test)))


# In[69]:


yo_pred = random_search.predict(x_test)
yo_pred


# In[70]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
conM = confusion_matrix(y_test,yo_pred)
conM
print(confusion_matrix(y_test,yo_pred))
print(classification_report(y_test,yo_pred))


# Here is the result after we optimized.The following were the correct classifications made by knn model 300 and 321. wrong classification were 27 and 38. In total we have 621 correct classifications and 65 wrong classifications. the algorithm had 91%. This is an improvement.

# In[71]:


print(random_search.best_params_)


# KNN performed best when the neighbor is set at 3

# ## 2. Decision Tree Model

# Let now make use of Decision Tree algorithm and assess it performance on our dataset

# In[72]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf


# In[73]:


## fit the Decision Tree model
clf.fit(x_train,y_train)


# In[74]:


clf.get_params()


# In[75]:


## Making Predictions with our Decision Tree Model
y_pred = clf.predict(x_test)
y_pred


# In[76]:


##Checking Accurarcy
print("Accurarcy is"), accuracy_score(y_test,y_pred)*100


# In[77]:


## Classification report
from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# With Decision Tree Algorithm we got a better result when compared to KNN result (before and after optimization).  Here is the result of decision tree, 328 and 327 were correct classifications. wrong classification were 21 and 10. In total we have 655 correct classifications and 31 wrong classifications. The Decision Tree algorithm had 95%. However, when the Decision Tree was optimized there was no improvement. The  macro average was still 95%.

# In[78]:


## Optimize
parameter = {'max_depth':(2,3,4,5,6,),
             'criterion': ('gini','entropy'),
             'min_samples_split':(2,4,6),
             'splitter' :('best', 'random')
             }


# In[79]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
grid = GridSearchCV(clf, param_grid = parameter, cv = 5, n_jobs = -1)
grid.fit(x_train,y_train)


# In[80]:


## Making Predictions with optimized Decision Tree Model
y_pred_grid = grid.predict(x_test)
y_pred_grid


# In[81]:


## Checking Accurarcy
print("Accurarcy is"), accuracy_score(y_test,y_pred_grid)*100


# In[82]:


## Classification report
from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(y_test,y_pred_grid)
print(confusion_matrix(y_test,y_pred_grid))
print(classification_report(y_test,y_pred_grid))


# In[83]:


grid.best_estimator_


# In[84]:


grid.best_score_


# In[85]:


grid.best_params_


# ## 3. Support Vector Machine

# In[86]:


##Building the Support Vector Machine (SVM)
from sklearn.svm import SVC
model_svm =SVC(random_state =123)
model_svm = SVC() 
model_svm


# In[87]:


## fit the SVM model
model_svm.fit(x_train,y_train)


# In[88]:


## svm Score
model_svm.score(x_test,y_test)


# In[89]:


## make prediction
y_pred = model_svm.predict(x_test)
y_pred


# In[90]:


## accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
acc


# In[91]:


## Confusion Matrix
from sklearn.metrics import confusion_matrix
svmcm = confusion_matrix(y_test,y_pred)
svmcm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[92]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
print ('Accuracy:', accuracy_score(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,
                              average='weighted'))
print ('Precision:', precision_score(y_test, y_pred,
                                    average='weighted'))
print ('\n clasification report:\n', classification_report(y_test, y_pred))
print ('\n confussion matrix:\n',confusion_matrix(y_test, y_pred))


# With Support Vector Machine the accuracy drop to 75% and the macro average was 74%. Comparing this with the earlier adopted machine learning algorithms that is the KNN and Decision Tree. The performance of Decision Tree is still the best which was 95% before and after optimization. Although when we optimized the support Vector Machine the accuarcy and macro average went up to 85% . This performance was not good enough to change position of Support Vector Machine admist these three algorithms. However Support Vector Machine had the best classification for "is not fraud" 333 correct classifications and 5 wrong classification. Here is the result of Support Vector Machine, 333 and 184 were correct classifications. wrong classification were 5 and 164. In total we have 517 correct classifications and 169 wrong classifications. The Decision Tree algorithm is still the bestwith 95%. 

# In[93]:


## Optimize Parameters with Cross validation and GridSearchCV
param_grid = { 'C': [0.5,1,10,100], 'gamma': ['scale',1,0.1,0.001,0.0001], 
               'kernel': ['rbf']}
optimal_params = GridSearchCV(SVC(), param_grid, cv=5,verbose = 0, n_jobs=-1)
optimal_params.fit(x_train, y_train)


# In[94]:


## svm Score
optimal_params.score(x_test,y_test)


# In[95]:


## make prediction
y_pred = optimal_params.predict(x_test)
y_pred


# In[96]:


## accuracy
from sklearn.metrics import accuracy_score
acc_op = accuracy_score(y_test,y_pred)
acc_op


# In[97]:


## Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
svmcmop = confusion_matrix(y_test,y_pred)
svmcmop
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[98]:


print(optimal_params.best_params_)


# ## 4. Multi-layer Perceptron alogrithms

# In[99]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=123)
print(x_train.shape)
print(x_test.shape) 


# In[100]:


#Train and fit the Multi-layer Perceptron Classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter = 500, activation = 'relu')
mlp.fit(x_train,y_train)


# In[101]:


#Predicting
predmlp = mlp.predict(x_test)
predmlp


# In[102]:


#Evaluation Metrics: Confusion Matrix and f2 Score
from sklearn.metrics import classification_report,confusion_matrix
cm_mlp = confusion_matrix(y_test,predmlp)
print(confusion_matrix(y_test,predmlp))
print(classification_report(y_test,predmlp))


# In[103]:


#Optimizing, Train the Multi-layer Perceptron Classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter = 500, activation = 'relu')
mlp.fit(x_train,y_train)


# In[104]:


#Predicting
y_pred = mlp.predict(x_test)
y_pred


# In[105]:


#Evaluation Metrics: Confusion Matrix and f2 Score
from sklearn.metrics import classification_report,confusion_matrix
cm_mlp = confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Out of 686, 610 were correctly classified while 76 were wrongly classified As stated above the accuracy is 89% is fraud  was the best classifications with 90% closely followed by is not fraud 88%. The accuracy and macro average were 89% respectively. Hoever, the result before optimization seems to be the best with accuracy and macro average being 93%. 296 and 343 were correct classifications 5 and 42 were wrong classifications. Note also that Multi-layer Perceptron algorithm has the best classifier for fraud.  Comparing the algorithms, Decision Tree 95%, Multi-layer Perceptron 93%, KNN 91% and Support vector 85%.

# ## 5. naiveBayes model

# In[106]:


# split data into train test data and fitting the model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state = 12)
from sklearn.naive_bayes import GaussianNB
GnB = GaussianNB()
GnB.fit(x_train,y_train)


# In[107]:


GnB.score(x_test,y_test)


# In[108]:


# making prediction with naive Bayes model
y_pred = GnB.predict(x_test)
y_pred


# In[109]:


#accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
acc


# In[110]:


#Confusion matrix and classification report
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# #Hyperparameter Tuning

# In[111]:


param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}


# In[112]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=5, n_jobs=-1)
nbModel_grid.fit(x_train, y_train)
print(nbModel_grid.best_estimator_)

#Fitting 5 folds 
GaussianNB(priors=None, var_smoothing=1.0)


# In[113]:


#making prediction
y_pred = nbModel_grid.predict(x_test)
print(y_pred)


# In[114]:


# Confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred), ": is the accuracy score")


# Discussion: The first time we run the Gaussian Naive Bayes Algorithm we got an accuracy score of 0.70 or 70%. We then do hyperparameter optimization. We perform a grid search which means we create a grid of possible values for hyperparameters. Each iteration tries a combination of hyperparameters in a specific order. It fits the model on each and every combination of hyperparameter possible and records the model performance. Finally, it returns the least model even when we conduct the  hyperparameters. In the function cv=5 which is a k-fold cross validation of 5. After that we predict on testing data then print the confusion matrix and accuracy score. After running the optimization process we now get an accuracy score of 0.70 or 70%.
