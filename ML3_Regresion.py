import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report #TO OUTPUT REPORT
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
##https://medium.com/@powersteh/an-introduction-to-applied-machine-learning-with-multiple-linear-regression-and-python-925c1d97a02b
###################data prepared#################################################################
##test

def ML3_Regression():

    X_train, X_test, y_train, y_test,targets,features,dataset = prepareData()


    '''
    ####remove outliers https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
    z = np.abs(stats.zscore(dataset))

    print("Z greater than 3")
    print(np.where(z>3))
    print(z[151][3]) # z over 3 example

    print("------------------------------------------")
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3-Q1
    print("IRQ FOR EACH COLUMN")
    print(IQR)
    IQR_OUTOUT = (dataset <(Q1 - 1.5*IQR)) | (dataset > (Q3+1.5*IQR))
    print(IQR_OUTOUT.shape)
    '''
## test Markdown
    ######################################################################
    #print("-----------------Stats of the data------------")
    #print(dataset.iloc[:,2:].describe())
    #stats = (dataset.iloc[:,2:].describe())

    #print("----------------------------------------------")

    linearRegressionRFE(features, targets,X_train, X_test, y_train, y_test)
    #try RFE squared https://www.datacamp.com/community/tutorials/feature-selection-python
    #####################################create Linear regression classifier################################
    linearRegression(features,targets,X_train, X_test, y_train, y_test)
    #create a Gradient Boosting Regressor####################################################################################
    gradientBoostingRegression(features,targets,X_train, X_test, y_train, y_test)
     #create a kNeighborRegression Regressor####################################################################################
    kNeighborRegression(features,targets,X_train, X_test, y_train, y_test)
     #create a PLSregression Regressor#################################################
    PLSregression(features,targets,X_train, X_test, y_train, y_test)
    #create a PLSregression Regressor#################################################
    SVM(features,targets,X_train, X_test, y_train, y_test)

def plotModel(expected2, predicted2):
    plt.figure(figsize=(4, 3))
    plt.scatter(expected2, predicted2)
    plt.plot([0, 500], [0, 500], '--k')
    plt.axis('tight')
    plt.xlabel('tensile_strength')
    plt.ylabel('Predicted tensile_strength')
    plt.tight_layout()
    plt.show()



def prepareData():
    filename = "steel.csv"
    dataset = pd.read_csv(filename,sep='\s+')# tab seperated
    print('Shape of input data::')
    print(dataset.shape)
    dataset = dataset[(np.abs(st.zscore(dataset)<3)).all(axis=1)] # get rig of out liers z greater than 3 = higher than average
    print('Shape of input data after outliers removed::')
    print(dataset.shape)##data reduce in size

 # Create the Scaler object##############################################

    #scaler = preprocessing.StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit your data on the scaler object
    dataset = scaler.fit_transform(dataset)
    dataset  = pd.DataFrame(dataset)

    #dataset = pd.DataFrame(preprocessing.scale(dataset))



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
    features = dataset.drop(['tensile_strength','percent_chromium','manufacture_year','percent_copper','percent_nickel','percent_sulphur'], axis=1)
    #'percent_nickel','percent_sulphur','percent_carbon','percent_manganese
    #features = dataset.drop('tensile_strength', axis=1)



    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=1) # 80% training and 20% test
    #sc_X = StandardScaler(features)
   # sc_Y = StandardScaler(targets)
    '''
    print('test data')
    print(X_train.shape)
    print(X_test.shape)
    print('train data')
    print(y_train.shape)
    print(y_test.shape)
    '''


    return X_train, X_test, y_train, y_test,targets,features,dataset

def linearRegression(features, targets,X_train, X_test, y_train, y_test):
    regression1 = LinearRegression()# multi v. linear regressor
    ##SELECT TOP 3 FEATUES TO USE.
    #print(features['manufacture_year'])

    rfe = RFE(regression1 ,6)
    FIT = rfe.fit(X_train, y_train)
   # print(X_train.columns)
    print(X_train.columns)



    print('Num Features:%s'%(FIT.n_features_))
    print("Features to pic:%s"%(FIT.support_))


    #Train classifier
    regression1.fit(X_train, y_train)
    predicted = regression1.predict(X_test)
    expected = y_test
    #now do 10 fold validation#########################################################################
    kfold1 = KFold(n_splits=10, random_state=100)
    #https://www.pluralsight.com/guides/validating-machine-learning-models-scikit-learn##################################
    results_kfold1 = cross_val_score(regression1,X_test,y_test, cv=kfold1)
    print('linearRegression run::')
    print("Accuracy for 10 fold validation for Linear Regressor: %.2f%%" % (results_kfold1.mean()*100.0))
    ##print("--------------------------------------------------------------" )
    #plot the input.
    ##https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
    ##https://medium.com/@powersteh/an-introduction-to-applied-machine-learning-with-multiple-linear-regression-and-python-925c1d97a02b
    print("RMS for Linear regressor: %r " % np.sqrt(np.mean((predicted - expected) ** 2)))
    print("--------------------------------------------------------------" )
    print("LR score", regression1.score(X_test,y_test))

def linearRegressionRFE(features,targets,X_train, X_test, y_train, y_test):
    #no of features
    nof_list=np.arange(1,13)
    high_score=0
    #Variable to store the optimum features
    nof=0
    score_list =[]
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(features,targets,test_size = 0.3, random_state = 0)
        model = LinearRegression()
        rfe = RFE(model,nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            nof = nof_list[n]

    print("LR with RFE run::")
    kfold2 = KFold(n_splits=10, random_state=100)
    results_kfold2 = cross_val_score(model, X_test_rfe,y_test, cv=kfold2)



    print("Accuracy for 10 fold validation linearRegressionRFE: %.2f%%" % (results_kfold2.mean()*100.0))
    #print("--------------------------------------------------------------" )
    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))
    print("--------------------------------------------------------------" )

def gradientBoostingRegression(features,targets,X_train, X_test, y_train, y_test):

    reg2 = GradientBoostingRegressor()

    rfe2 = RFE(reg2 ,6)

    FIT2 = rfe2.fit(X_train, y_train)
    print('gradientBoostingRegression run::')
    print('Num Features:%s'%(FIT2.n_features_))
    print("Features to pic:%s"%(FIT2.support_))
    #train it
   # print(X_train.columns)

    #features = features.drop(['sample','percent_manganese','percent_silicon'], axis=1)
    reg2.fit(X_train, y_train)

    predicted2 = reg2.predict(X_test)
    expected2 = y_test

    kfold2 = KFold(n_splits=10, random_state=100)
    results_kfold2 = cross_val_score(reg2, features, targets, cv=kfold2)

    print("Accuracy for 10 fold validation Gradient Boosting Regressor: %.2f%%" % (results_kfold2.mean()*100.0))
    #print("--------------------------------------------------------------" )

    ##https://shankarmsy.github.io/stories/gbrt-sklearn.html
    print("RMS for Gradient Boosting Regressor: %r " % np.sqrt(np.mean((predicted2 - expected2) ** 2)))
    ##print("ski RMS", np.sqrt(mt.mean_squared_error(expected2,predicted2)))
    print("--------------------------------------------------------------" )
    #https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
    #https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_boston_prediction.html
    #print("SVM RMS: %r " % np.sqrt(np.mean((predicted4 - expected4) ** 2)))
def kNeighborRegression(features,targets,X_train, X_test, y_train, y_test):
    reg3 = KNeighborsRegressor(n_neighbors = 4)

    reg3.fit(X_train , y_train)

    predicted3 = reg3.predict(X_test)
    expected3 = y_test
    pointPredictions = reg3.predict(X_train)
    #print(pointPredictions)
    kfold2 = KFold(n_splits=10, random_state=100)
    results_kfold3 = cross_val_score(reg3, features, targets, cv=kfold2)


    print('kNeighborRegression run::')
    print("Accuracy for 10 fold validation on KNN: %.2f%%" % (results_kfold3.mean()*100.0))

    print("RMS for KNN: %r " % np.sqrt(np.mean((predicted3 - expected3) ** 2)))

    print("--------------------------------------------------------------" )


def PLSregression(features,targets,X_train, X_test, y_train, y_test):
    reg4 = PLSRegression(n_components=5)
    #train it
    #rfe = RFE(reg4,9)
    #X_rfe = rfe.fit_transform(X_train, y_train)

    #reg4.fit(X_rfe , y_train)
    reg4.fit(X_train,y_train)

    #predicted4 = reg4.predict(X_test)
    #expected4 = y_test

    y_cv = cross_val_predict(reg4,X_test,y_test,cv=10)
    r_score = r2_score(y_test,y_cv)
    mse = mean_squared_error(y_test,y_cv)
    kfold2 = KFold(n_splits=10, random_state=100)
    results_kfold3 = cross_val_score(reg4, features, targets, cv=kfold2)
    print('PLSregression run::')
    print("r2 score for PLS Regressor",r_score)
    print("Accuracy for 10 fold validation on PLS Regressor: %.2f%%" % (results_kfold3.mean()*100.0))
    print("mean Squared Error score for PLS Regressor",mse)
    #print("Accuracy for 10 fold validation for SVM: %.2f%%" % (results_kfold4.mean()*100.0))
    print("--------------------------------------------------------------" )
##https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d
def SVM(features,targets,X_train, X_test, y_train, y_test):
    #reg4 = SVR(kernel='linear')
    reg4 = SVR(gamma = 'auto',kernel='rbf')


    reg4.fit(X_train,y_train)

    #predicted4 = reg4.predict(X_test)
    #expected4 = y_test
    #results_kfold4 = cross_val_score(reg4, features1, targets1, cv=kfold2)
    y_cv = cross_val_predict(reg4,X_test,y_test,cv=10)
    r_score = r2_score(y_test,y_cv)
    mse = mean_squared_error(y_test,y_cv)
    kfold2 = KFold(n_splits=10, random_state=100)
    results_kfold3 = cross_val_score(reg4, features, targets, cv=kfold2)
    print('SVM regression run::')
    print("r2 score for SVM Regressor",r_score)
    print("Accuracy for 10 fold validation on SVM Regressor: %.2f%%" % (results_kfold3.mean()*100.0))
    print("mean Squared Error score for SVM Regressor",mse)
    #print("Accuracy for 10 fold validation for SVM: %.2f%%" % (results_kfold4.mean()*100.0))
    print("--------------------------------------------------------------" )

def main():
    '''
    function def::
    Calls the regression algorithms.
    '''
    ML3_Regression()

if __name__=="__main__":#execute only if run as a script
    main()
