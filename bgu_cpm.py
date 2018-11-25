import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer


def create_x(df, is_train):
    # fill mising value with net valid sample
    df.fillna(method='bfill', inplace=True)
    # make train X and y , TODO : make to split for CV
    if is_train:
        X_train = df.drop(['Class'], axis=1)
        y_train = df.Class
    else:
        X_train = df
    # categorial to thire cat code
    obj_df = X_train.select_dtypes(include=['object']).copy()
    category = list(obj_df.columns)
    for col in category:
        X_train[col] = obj_df[col].astype('category')
        X_train[col] = X_train[col].cat.codes
    # fill NaN value with the mean of respective column
    X_train.fillna(X_train.mean(), inplace=True)
    if is_train:
        y_train.fillna(1, inplace=True)  # just temporary put 1 calcification for missing values
        return X_train, y_train
    else:
        return X_train, 0


rf = RandomForestRegressor(n_estimators=1000, random_state=42)

train_df = pd.read_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\saftey_efficay_myopiaTrain.csv')
test_df = pd.read_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\saftey_efficay_myopiaTest.csv')

# drop samples with no classification
train_df.dropna(subset=['Class'], inplace= True)

X_train, y_train = create_x(train_df, 1)

X_test, dummy = create_x(test_df, 0)

print("start fiting the model")
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("after fit #############")
print("feature_importances_ : {}".format(rf.feature_importances_))

y_pred.to_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\submission.csv', sep=',', index = False)
