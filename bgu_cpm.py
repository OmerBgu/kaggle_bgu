import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

train_df = pd.read_csv(r'C:\Users\omer_an\Downloads\kaggle_bgu\all\saftey_efficay_myopiaTrain.csv')
#train_df.dropna(inplace = True)
train_df.fillna(method='bfill',inplace=True)
print("{}{}".format(train_df.head(5), train_df.describe()))
X_train = train_df.loc[:, train_df .columns != 'Class']
y_train = train_df.loc[:, ~train_df.columns.isin(['Class'])]

cleanup_nums = {"D_L_Sex":     {"F": 1, "M": 0},
                "D_L_Eye": {"Right": 1, "Left": 0},
                "T_L_Therapeutic_Cont._L.":     {"Yes": 1, "No": 0}}
X_train.replace(cleanup_nums, inplace=True)

#print(train_df.isnull().sum())

col_name = X_train.select_dtypes(include=['object']).copy().columns.values
print(col_name)


# values = X_train .values
# imputer = Imputer()
# cat_features = ['D_L_Sex', 'D_L_Eye', 'D_L_Dominant_Eye' 'Pre_L_Contact_Lens'
#   'T_L_Laser_Type' 'T_L_Treatment_Type' 'T_L_Cust._Ablation' 'T_L_Micro'
#   'T_L_Head' 'T_L_Therapeutic_Cont._L.' 'T_L_Epith._Rep.']
# enc = LabelEncoder()
# enc.fit(cat_features)
# new_cat_features = enc.transform(cat_features)
# print(new_cat_features)
# new_cat_features = new_cat_features.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False) #Easier to read
# print(ohe.fit_transform(new_cat_features))
#
# enc = OneHotEncoder(categorical_features=cat_features)
# enc.fit(X_train.values)
#
# label_encoder = LabelEncoder()
# for col in col_name:
#     print(col)
#     X_train[col] = label_encoder.fit_transform(X_train[col])
#
#
# print("after convert{}{}".format(X_train.dtypes, X_train.head()))
#
# enc = OneHotEncoder(handle_unknown='ignore')
# enc.fit(X_train)
# print(enc.categories_)
#print("****not null : {}".format(X_train[X_train.isnull()]))

# print("*********T_L_Epith._Rep. watch : {}".format(X_train['T_L_Epith._Rep.'].isnull().sum()))
cat_df_onehot = X_train.copy()
execpt_yes_no = ['T_L_Epith._Rep.', 'T_L_Therapeutic_Cont._L.']
col_name_list = list(col_name)
for col in execpt_yes_no :
    cat_df_onehot= pd.get_dummies(cat_df_onehot, columns=[col], prefix = [col])
    X_train= cat_df_onehot
    col_name_list.remove(col)


for l in col_name_list:
    print("handle {}".format(l))
    X_train[l] = X_train[l].astype('category')
    X_train[l] = X_train[l].cat.codes

rf = RandomForestRegressor(n_estimators=1000, random_state=42)
print("--------------before fit")


rf.fit(X_train, y_train)
print("--------------after fit")