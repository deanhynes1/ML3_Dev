import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report #TO OUTPUT REPORT
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
from sklearn import metrics
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt


def ML3_Regression():
    X_train, X_test, y_train, y_test,targets,features,dataset = prepareData()
    #call linar regression and svm.
    linearRegression(features,targets,X_train, X_test, y_train, y_test)
    SVM(dataset)#


def plotModel(target, input):# used to check each feature againts the target
    plt.scatter(target, input,color='green')
    plt.plot([-1, 2], [-1, 2], '--k')
    plt.grid(True)
    plt.show()

def prepareData():#read the csv and scale the data+ add feature+targets,
    filename = "steel.csv"
    dataset = pd.read_csv(filename,sep='\s+')# tab seperated
 # Create the Scaler object##############################################
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit data on the scaler object
    dataset = scaler.fit_transform(dataset)
    dataset  = pd.DataFrame(dataset)

    dataset.columns =['normalising_temperature','tempering_temperature', 'sample_id', 'percent_silicon', 'percent_chromium', 'manufacture_year', 'percent_copper', 'percent_nickel','percent_sulphur','percent_carbon','percent_manganese','tensile_strength']
    #Features ranking:[       4                       5                     6            1                  3                     7                  1                1                   1                1                   2]
    #Features to pic:[            True                True                 True               True        False                False                False              False            False             True                 True
    ## Split the data into features and target
    #do we need sample, manufacture_year,normalising_temperature',

    #thisYear = 2019
    #dataset['manufacture_year'] = thisYear - dataset['manufacture_year']
    #print(dataset[ 'sample'])
    print("------------------------------------------")
    '''
    ##should all give false to say no data is missing, which it does.
    print(pd.isnull(dataset).any())# no missing value check.
    print("------------------------------------------")
    '''
    targets = dataset["tensile_strength"]# id the target
    #features = dataset.drop("tensile_strength", axis=1)   # Drop the variety name since this is our target
    #features = dataset.drop(['tensile_strength','percent_manganese','tempering_temperature','percent_sulphur'], axis=1)
    features = dataset.drop(['tensile_strength','percent_chromium','manufacture_year','percent_copper','percent_nickel','percent_sulphur'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=1) # 70% training and 20% test
    #dataset.columns =['normalising_temperature','tempering_temperature', 'sample_id', 'percent_silicon', 'percent_chromium', 'manufacture_year', 'percent_copper', 'percent_nickel','percent_sulphur','percent_carbon','percent_manganese','tensile_strength']
   #                            l                          nl                   wl           wl               wl                   wl                 wl                 wl               nl              wl                nl
          #'percent_chromium','manufacture_year','percent_copper','percent_nickel','percent_sulphur'], axis=1)

    #plotModel(targets, dataset['percent_manganese'])
    return X_train, X_test, y_train, y_test,targets,features,dataset
#linearRegression model
def linearRegression(features, targets,X_train, X_test, y_train, y_test):
    regression1 = LinearRegression(n_jobs=-1)# multi v. linear regressor
    ##SELECT TOP 3 FEATUES TO USE.
    #print(features['manufacture_year'])

    rfe = RFE(regression1 ,6)#check best features of the data.
    FIT = rfe.fit(X_train, y_train)
    #print('Num Features:%s'%(FIT.n_features_))
    #print("Features to pic:%s"%(FIT.support_))
    #Train classifier
    regression1.fit(X_train, y_train)
    predicted = regression1.predict(X_test)
    expected = y_test
    r_score = r2_score(y_test,predicted)
    adr_r_score = 1 - (1-r_score)*(len(features)-1)/(len(features)-(features.shape[1]-1)-1)
    mse = mean_squared_error(y_test,predicted)
    #now do 10 fold validation#########################################################################
    kfold1 = KFold(n_splits=10, random_state=100)
    results_kfold1 = cross_val_score(regression1,X_test,y_test, cv=kfold1)
    print('linearRegression run::')
    #print("Training set accuracy:,{:.2f}".format(regression1.score(X_train,y_train)))
    print("Accuracy for 10 fold validation for Linear Regressor: %.2f%%" % (results_kfold1.mean()*100.0))
    print("R2 score for Linear Regression:",r_score )
    print("Adjusted Rscore for Linear regressor:",adr_r_score)
    print("LR score:", regression1.score(X_test,y_test))
    print("RMSE score for Linear regressor:",sqrt(mse))

    print("--------------------------------------------------------------" )
#svm model
def SVM(dataset):
    #manually check kernels
    #reg4 = SVR(kernel='linear')
    reg4 = SVR(gamma = 'auto',kernel='rbf',epsilon=.05)
    #reg4 = SVR(gamma = 'auto',C=100,kernel='linear')
    #reg4 = SVR(gamma = 'auto',kernel='poly',C=100,degree=3,epsilon=.1,coef0=1)
    dataset = dataset[(np.abs(st.zscore(dataset)<2)).all(axis=1)]
    targets = dataset["tensile_strength"]# id the target    #features = dataset.drop("tensile_strength", axis=1)   # Drop the variety name since this is our target    #features = dataset.drop(['tensile_strength','percent_manganese','tempering_temperature','percent_sulphur'], axis=1)
    features = dataset.drop(['tensile_strength','percent_chromium','manufacture_year','percent_copper','percent_nickel','percent_sulphur'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=1)
    reg4.fit(X_train,y_train)
    y_pred = reg4.predict(X_test)
    y_cv = cross_val_predict(reg4,X_test,y_test,cv=10)

    r_score = r2_score(y_test,y_cv)
    adr_r_score = 1 - (1-r_score)*(len(features)-1)/(len(features)-(features.shape[1]-1)-1)
    mse = mean_squared_error(y_test,y_cv)
    kfold2 = KFold(n_splits=10, random_state=100)
    results_kfold3 = cross_val_score(reg4, features, targets, cv=kfold2)
    print('SVM regression run::')
    print("Accuracy for 10 fold validation on SVM Regressor: %.2f%%" % (results_kfold3.mean()*100.0))
    print("R2 score for SVM Regressor",r_score)
    print("LR score", reg4.score(X_test,y_test))
    print("Adjusted Rscore for SVM Regressor",adr_r_score)
    print("RMSE score for SVM Regressor",sqrt(mse))
    #print("Accuracy for 10 fold validation for SVM: %.2f%%" % (results_kfold4.mean()*100.0))

    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("--------------------------------------------------------------")

def main():
    '''
    function def::
    Calls the regression algorithms.
    '''
    ML3_Regression()

if __name__=="__main__":#execute only if run as a script
    main()
