import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import csv
from sklearn.metrics import roc_auc_score
import os

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


def create_x(df, is_train):
    # fill mising value with net valid sample
    df.fillna(method='bfill', inplace=True) # maybe test ffik or backfill
    if is_train:
        X_train = df.drop(['Class'], axis=1)
        y_train = df.Class
    else:
        X_train = df
    # categorial to thire cat code
    obj_df = X_train.select_dtypes(include=['object']).copy()
    category = list(obj_df.columns)
    for col in category:
        # each categorial get dummy collumn
        X_train[col] = pd.get_dummies(X_train[col], columns=[col], prefix=[col])
        dummy = pd.get_dummies(X_train[col])
        X_train = X_train.drop([col], axis=1)
        X_train = pd.concat([X_train,dummy], axis=1)

        #X_train[col] = obj_df[col].astype('category')
        #X_train[col] = X_train[col].cat.codes
    # fill NaN value with the mean of respective column
    #X_train.fillna(X_train.mean(), inplace=True)
    X_train.interpolate(inplace=True)
    if is_train:
        y_train.interpolate(inplace=True)
        y_train.fillna(1, inplace=True)
        return X_train, y_train
    else:
        return X_train, 0

submision_apth =r'C:\Users\omera\Downloads\kaggle_bgu\all\submission_with_tunning.csv'
base_path = r'C:\Users\omera\Downloads\kaggle_bgu\all\\'
if __name__ == '__main__':


    if os.path.exists(submision_apth ):
        os.remove(submision_apth )

    rf = RandomForestRegressor(n_estimators=500,max_features=9, min_samples_split=8, min_samples_leaf=4, max_depth=100,bootstrap=True, random_state=42)

    # {'bootstrap': True, 'max_depth': 60, 'max_features': 9, 'min_samples_leaf': 4, 'min_samples_split': 8}
    #n_estimators': 200, min_samples_split: 2, min_samples_leaf: 4, max_depth: 40, bootstrap: True
    #{'bootstrap': True, 'max_depth': 100, 'max_features': 9, 'min_samples_leaf': 4, 'min_samples_split': 8, 'n_estimators': 500}

    #{'max_depth': 8, 'max_features': 'auto', 'n_estimators': 500}

    train_df = pd.read_csv(base_path  + r'saftey_efficay_myopiaTrain.csv')
    test_df = pd.read_csv( base_path  + r'saftey_efficay_myopiaTest.csv')

    # drop samples with no classification
    train_df.dropna(subset=['Class'], inplace=True)

    X_train, y_train = create_x(train_df, 1)

    X_competiton, dummy = create_x(test_df, 0)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
    #sm = SMOTE(random_state=42, ratio=0.98)
    sm = SMOTE(random_state=42,ratio=0.35) # 0.4 -> 0.64568 , 0.5 - > 0.64228, 0.3 - > 0.64342, 0.5 - >  0.65504 (best so far)
    X_train, y_train = sm.fit_sample(X_train, y_train)

    look_for_best_param = False
    if look_for_best_param:
        param_grid = {
            'bootstrap': [True],
            'max_depth': [ 60,80,100, 150],
            'max_features': [5, 7, 9,'auto'],
            'min_samples_leaf': [4, 7, 15],
            'min_samples_split': [8, 10, 12]
            #'n_estimators': [300, 500 ,1000]
        }

        # param_grid = {
        #     'n_estimators': [200, 500,700],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [4, 5, 6, 7, 8]
        # }


        rf_random = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 1)

        print("start grisdSearchCV")
        rf_random.fit(X_train, y_train)

        print("best params are : \n{} ".format(rf_random.best_params_)) #{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_depth': 40, 'bootstrap': True}
        exit(1)

    print("start fitting the model")
    rf.fit(X_train, y_train)
    # base_accuracy_no_tuning = evaluate(rf, X_test, y_test)
    # base_accuracy_with_tuning = evaluate(rf_random, X_test, y_test)
    # print("base_accuracy_no_tuning  : {} and base_accuracy_with_tuning  : {} ".format(base_accuracy_no_tuning ,
    #                                                                                   base_accuracy_with_tuning ))

    # y_pred_test = rf.predict(X_test)
    # auc = roc_auc_score(y_test, y_pred_test)
    # print("auc score on train set : {}".format(auc))

    y_pred_no_tune = rf.predict(X_competiton)
    y_pred_no_tune = y_pred_no_tune.astype(float)
    mean = np.mean(y_pred_no_tune)
    y_pred_no_tune = y_pred_no_tune.astype(float)
    # y_pred_with_tune = rf_random.predict(X_competitopn)
    print("end")

    with open(base_path + r'submission_with_tunning.csv', 'w', newline='') as csvfile:
        indices = list(range(1, len(y_pred_no_tune) + 1))
        fieldnames = ["Id", "Class"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(0, len(y_pred_no_tune)):
            if y_pred_no_tune[i] != 0:
                writer.writerow({'Id': indices[i], 'Class': y_pred_no_tune[i]})
            else:
                writer.writerow({'Id': indices[i], 'Class': y_pred_no_tune[i]})
                #writer.writerow({'Id': indices[i], 'Class': mean})


