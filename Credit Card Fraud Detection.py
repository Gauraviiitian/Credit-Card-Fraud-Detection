
# coding: utf-8

# In[10]:


import sys
import matplotlib as mpl
import numpy as np 
import pandas as pd
import seaborn as sb
import scipy as sp
import sklearn

print "Python: {}".format(sys.version)
print "Numpy: {}".format(np.__version__)
print "Pandas: {}".format(pd.__version__)
print "Scipy: {}".format(sp.__version__)
print "Seaborn: {}".format(sb.__version__)
print "Matplotlib: {}".format(mpl.__version__)
print "Sklearn: {}".format(sklearn.__version__)


# In[12]:


#import the necessary packages
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
import scipy as sp


# In[15]:


#Load the dataset from the .csv file using pandas
data = pd.read_csv("creditcard.csv")


# In[19]:


#Explore the dataset
print data.columns


# In[20]:


print data.shape


# In[21]:


print data.describe()


# In[22]:


data = data.sample(frac = 0.1, random_state = 1)

print data.shape


# In[23]:


#plot histogram of each parameter
data.hist(figsize = (20, 20))
plt.show()


# In[24]:


#Determine number of fraud and valid cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
print outlier_fraction

print "Fraud Cases: {}".format(len(fraud))
print "Valid Cases: {}".format(len(valid))


# In[25]:


#Correlation Matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[26]:


#Get all the columns from the dataframe
columns = data.columns.tolist()

#filter the columns to remove data we donot want
columns = [c for c in columns if c not in ["Class"]]

#Store the variable we'll be predicting on 
target = "Class"

X = data[columns]
Y = data[target]

#print the shapes of X and Y
print X.shape
print Y.shape


# In[31]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state
state = 1
#define the outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples = len(X),
                                       contamination = outlier_fraction,
                                       random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(
    n_neighbors=20,
    contamination = outlier_fraction)
}


# In[36]:


#fit the model
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    #Reshape the prediction value to 0 for valid and 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    #Run Classification Metrics
    print "{}: {}".format(clf_name, n_errors)
    print accuracy_score(Y, y_pred)
    print classification_report(Y, y_pred)

