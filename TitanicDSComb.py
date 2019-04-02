# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#This was code I wrote for a Kaggle competition to predict survivorship on the Titanic.
#Mathew Strassburg February 1, 2019

#Import libraries
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt #matplotlib inline


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Import training and test data
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

dataset = pd.concat([train_df, test_df]) #combine datasets so feature engineering is applied to both

# preview the data
print(train_df.columns.values)
train_df.head()
train_df.tail()

#train_df.info()
#print('_'*40)
#test_df.info()
#train_df.describe()

#Explore the data to see what features matter---------------------------------
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

#print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
#print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


#Create Title feature----------------------------------------------------------

dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

dataset['Title'] = dataset['Title'].map(title_mapping)
dataset['Title'] = dataset['Title'].fillna(0)


dataset = dataset.drop(['Name', 'PassengerId'], axis=1)

dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


#Fill Age values and create bands----------------------------------------------
guess_ages = np.zeros((2,6))

for i in range(0, 2):
    for j in range(0, 6):
        guess_df = dataset[(dataset['Sex'] == i) & (dataset['Title'] == j)]['Age'].dropna()

        # age_mean = guess_df.mean()
        # age_std = guess_df.std()
        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
        
        if len(guess_df) != 0:
            age_guess = guess_df.median()
        else: age_guess = 0

        # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
#        print(i,j,age_guess)
        
#Fill in blank ages with median ages from above bucketing.        
for i in range(0, 2):
    for j in range(0, 6):
        dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Title == j),\
                'Age'] = guess_ages[i,j]

dataset['Age'] = dataset['Age'].astype(int)

#Bucket ages to convert to categorical feature type
dataset['AgeBand'] = pd.cut(dataset['Age'], 5)
dataset[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

dataset = dataset.drop(['AgeBand'], axis=1)


#Create new variable for Family size-------------------------------------------

dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

dataset['IsAlone'] = 0
dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

dataset = dataset.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

dataset['Age*Class'] = dataset.Age * dataset.Pclass


#Fix embarked------------------------------------------------------------------
freq_port = dataset.Embarked.dropna().mode()[0]

dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#Fix the fares-----------------------------------------------------------------
dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

#Find bucket limits for fares
dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)
dataset[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
dataset['Fare'] = dataset['Fare'].astype(int)

dataset = dataset.drop(['FareBand'], axis=1)

#Run the ML--------------------------------------------------------------------
X_train = dataset.iloc[1:len(train_df)].drop("Survived", axis=1)
Y_train = dataset.iloc[1:len(train_df)].Survived
X_test  = dataset.iloc[len(train_df):].drop("Survived", axis=1)
X_train.shape, Y_train.shape, X_test.shape

#Run
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("acc_log ", acc_log)

#Correlations
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print("acc_svc ", acc_svc)

#K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("acc_knn ", acc_knn)

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print("acc_gaussian ", acc_gaussian)

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print("acc_perceptron ", acc_perceptron)

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print("acc_linear_svc ",acc_linear_svc)


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print("acc_sgd ", acc_sgd)

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("acc_decision_tree ", acc_decision_tree)

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("acc_random_forest ", acc_random_forest)



#Grid Search Random Forest (these steps are supposed to autotune parameters but we don't see improvement)
#
#X, y = make_classification(n_samples=891,
#                           n_features=8,
#                           n_informative=3,
#                           n_redundant=0,
#                           n_repeated=0,
#                           n_classes=2,
#                           random_state=0,
#                           shuffle=False)


rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = False) 

param_grid = { 
    'n_estimators': [10,23,30],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [5, 10, 30]
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, Y_train)
print(CV_rfc.best_params_)
CV_rfc.score(X_train, Y_train)
pred = CV_rfc.predict(X_test)
acc_cv_random_forest = round(CV_rfc.score(X_train, Y_train) * 100, 2)
print("acc_cv_random_forest ", acc_cv_random_forest)

#Score Models
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

#------Random forest and Decision tree models win with 87% accuracy of predicting survival on test set--------


#Save file
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred
    })
    
submission.to_csv('../submissionpy.csv', index=False)