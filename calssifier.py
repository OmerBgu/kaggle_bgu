import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA

#from RotationForest.RotationForest import RotationForest


def create_x(df, is_train):
    # fill mising value with net valid sample
    df.fillna(method='bfill', inplace=True)
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


submission_path =r'C:\Users\omera\Downloads\kaggle_bgu\all\submission_with_tunning.csv'
base_path = r'C:\Users\omera\Downloads\kaggle_bgu\all\\'

if __name__ == '__main__':

    if os.path.exists(submission_path ):
        os.remove(submission_path )

    rf = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=4, max_depth=80,
                               bootstrap=True, random_state=42) # 0.5813121361665429

    train_df = pd.read_csv(base_path + r'saftey_efficay_myopiaTrain.csv')
    test_df = pd.read_csv(base_path + r'saftey_efficay_myopiaTest.csv')

    # drop samples with no classification
    train_df.dropna(subset=['Class'], inplace=True)

    X_train, y_train = create_x(train_df, 1)

    X_competitopn, dummy = create_x(test_df, 0)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    look_for_best_param = False
    if look_for_best_param:
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

    # TODO : this is (GB) not working
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    param_test1 = {
        'loss' : {'ls', 'lad', 'huber', 'quantile'},
        'learning_rates': list([1, 0.5, 0.25, 0.1, 0.05, 0.01])
        #'max_depth': max_depth
    }
    #PCA
    pca = PCA(n_components=len(X_train.columns), whiten=False, copy=True)

    principalComponents = pca.fit_transform(X_train)
    #print(principalComponents)
    print(pca.explained_variance_ratio_)

    # Explained variance
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    #plt.show()


    print("start fitting RF the model")
    rf.fit(X_train, y_train)


    y_pred_test = rf.predict_log_proba(X_test)
    #auc = roc_auc_score(y_test, y_pred_test)
    #print("RF aux test : {}".format(auc))

    y_pred_no_tune = rf.predict(X_competitopn)
    mean = np.mean(y_pred_no_tune)
    y_pred_no_tune = y_pred_no_tune.astype(float)
    # y_pred_with_tune = rf_random.predict(X_competitopn)
    print("after fit #############")

    # print importance fuatures by RF
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    labels = list(X_train.columns)
    for f in range(12):
        print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], labels[f]))

    indices = indices[0:12]
    x_col = 12
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x_col), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(x_col), labels, fontsize=6)

    plt.xlim([-1, x_col])

    show_plot = 0
    if show_plot:
        plt.show()

    with open( base_path + r'submission_with_tunning.csv', 'w', newline='') as csvfile:
        indices = list(range(1, len(y_pred_no_tune)+1))
        fieldnames = ["Id", "Class"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(0, len(y_pred_no_tune)):
            if y_pred_no_tune[i] != 0:
                writer.writerow({'Id': indices[i], 'Class': y_pred_no_tune[i]})
            else:
                writer.writerow({'Id': indices[i], 'Class': mean})
    #prediction = pd.DataFrame(y_pred_no_tune, columns=['predictions']).to_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\submission_with_tunning.csv',header=["Class"])
    # prediction = pd.DataFrame(y_pred_with_tune, columns=['predictions']).to_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\submission_with_tunning.csv',header=["Id","Class"])
