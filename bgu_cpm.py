import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split


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


if __name__ == '__main__':
    y_pred_no_tune = np.ndarray(shape=(2,2), dtype=float, order='F')
    prediction = pd.DataFrame(y_pred_no_tune, columns=['predictions']).to_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\test_my.csv', header=["Id","Class"])
    exit(1)
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    train_df = pd.read_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\saftey_efficay_myopiaTrain.csv')
    test_df = pd.read_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\saftey_efficay_myopiaTest.csv')

    # drop samples with no classification
    train_df.dropna(subset=['Class'], inplace=True)

    X_train, y_train = create_x(train_df, 1)

    X_competitopn, dummy = create_x(test_df, 0)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    # hyper tunning of parameters of RF
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
     #              'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    print(random_grid)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2,
                                   random_state=42, n_jobs=-1)

    print("start RandomizedSearchCV")
    rf_random.fit(X_train, y_train)

    print("best params are : \n{} ".format(rf_random.best_params_)) #{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_depth': 40, 'bootstrap': True}


    print("start fitting the model")
    rf.fit(X_train, y_train)
    base_accuracy_no_tuning = evaluate(rf, X_test, y_test)
    base_accuracy_with_tuning = evaluate(rf_random, X_test, y_test)
    print("base_accuracy_no_tuning  : {} and base_accuracy_with_tuning  : {} ".format(base_accuracy_no_tuning ,
                                                                                      base_accuracy_with_tuning ))

    y_pred_no_tune = rf.predict(X_competitopn)
    y_pred_with_tune = rf_random.predict(X_competitopn)
    print("after fit #############")

    prediction = pd.DataFrame(y_pred_no_tune, columns=['predictions']).to_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\submission_no_tunning.csv',header=["Id","Class"])
    prediction = pd.DataFrame(y_pred_with_tune, columns=['predictions']).to_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\submission_with_tunning.csv',header=["Id","Class"])

    #todo : get this erro with submission :Evaluation Exception:  is not a valid value for Double. Parameter name: value
    # chagne predicted values to double
    # get auc real score