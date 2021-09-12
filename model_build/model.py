# -*- coding: utf-8 -*-
"""model.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

df=pd.read_csv("online_shoppers_intention.csv")
df.head(5)


# # Column Descriptions:

# Administrative: This is the number of pages of this type (administrative) that the user visited.
#
# Administrative_Duration: This is the amount of time spent in this category of pages.
#
# Informational: This is the number of pages of this type (informational) that the user visited.
#
# Informational_Duration: This is the amount of time spent in this category of pages.
#
# ProductRelated: This is the number of pages of this type (product related) that the user visited.
#
# ProductRelated_Duration: This is the amount of time spent in this category of pages.
#
# BounceRates: The percentage of visitors who enter the website through that page and exit without triggering any additional tasks.
#
# ExitRates: The percentage of pageviews on the website that end at that specific page.
#
# PageValues: The average value of the page averaged over the value of the target page and/or the completion of an eCommerce transaction.
# More information about how this is calculated
#
# SpecialDay: This value represents the closeness of the browsing date to special days or holidays (eg Mother's Day or Valentine's day) in which the transaction is more likely to be finalized. More information about how this value is calculated below.
#
# Month: Contains the month the pageview occurred, in string form.
#
# OperatingSystems: An integer value representing the operating system that the user was on when viewing the page.
#
# Browser: An integer value representing the browser that the user was using to view the page.
#
# Region: An integer value representing which region the user is located in.
#
# TrafficType: An integer value representing what type of traffic the user is categorized into.
# Read more about traffic types here.
#
# VisitorType: A string representing whether a visitor is New Visitor, Returning Visitor, or Other.
#
# Weekend: A boolean representing whether the session is on a weekend.
#
# Revenue: A boolean representing whether or not the user completed the purchase.



df.info()

df.nunique()

df.VisitorType.unique()
df.PageValues.unique()

df.drop(columns=["Region","TrafficType","OperatingSystems","Administrative_Duration","Informational_Duration","Browser","BounceRates","ExitRates"],axis=1,inplace=True)

df.head()

df["Revenue"]=df["Revenue"].astype(int) 
df["Weekend"]=df["Weekend"].astype(int) 
df.head()

df = pd.get_dummies(df, columns=[ "Month","VisitorType"], drop_first=True)
df.head()

revenue_column=df['Revenue']
df.drop("Revenue",inplace=True,axis=1)

X = df.iloc[:, :].values
y = revenue_column.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#pip install catboost

from catboost import CatBoostClassifier

clf = CatBoostClassifier(
    iterations=500, 
    learning_rate=0.1, 
    depth=4,
    l2_leaf_reg=5,
    loss_function='CrossEntropy'
)
clf.fit(X_train, y_train, eval_set=(X_test, y_test),)

clf.score(X_test,y_test)

import pickle
pickle.dump(clf, open('shoppers.pkl', 'wb'))
