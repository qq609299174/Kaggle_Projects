XGB_WEIGHT = 0.6415
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0828
BASELINE_PRED = 0.0115
XGB1_WEIGHT = 0.8083   # first of the 2 XGB models combine together

print(XGB_WEIGHT, BASELINE_WEIGHT, OLS_WEIGHT, BASELINE_PRED)

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import random
from datetime import datetime
import gc
import pandas as pd

train2 = pd.read_csv('C:/Users/J/Documents/properties_2016.csv')
train1 = pd.read_csv('C:/Users/J/Documents/train_2016_v2.csv')

for c, dtype in zip(train2.columns, train2.dtypes):
    if dtype == np.float64:
        train2[c] == train2[c].astype(np.float32)

df_train = train1.merge(train2, how='left', on='parcelid')
df_train.fillna(df_train.median(), inplace=True)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train;
gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'  # mae
params['sub_feature'] = 0.5
params['bagging_fraction'] = 0.85
params['bagging_freq'] = 40
params['num_leaves'] = 512
params['min_data'] = 500
params['min_hessian'] = 0.05
params['verbose'] = 0

print("\nFitting Lightgbm model...")
clf = lgb.train(params, d_train, 430)

del d_train;
gc.collect()
del x_train;
gc.collect()

print("\nPrepare for Lightgbm prediction...")
print("Read sample file...")
sample = pd.read_csv('C:/Users/J/Documents/sample_submission.csv')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(train2, on='parcelid', how='left')
print("Merge done")
del sample, train2;
gc.collect()

x_test = df_test[train_columns]
del df_test;
gc.collect()

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart Lightgbm prediction...")

clf.reset_parameter({"num_threads": 1})  # thread num greater than 1 will run slowly
p_test = clf.predict(x_test)

del x_test;
gc.collect()

print("\nUnadjusted Lightgbm prediction:")
print(pd.DataFrame(p_test).head())

# Prepare for Xgboost
properties = pd.read_csv('C:/Users/J/Documents/properties_2016.csv')

for c in properties.columns:
    properties[c] = properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train1.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)

print('Shape train:{}\nShape test:{}'.format(x_train.shape, x_test.shape))

train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.42]
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print("Outliers removed:")
print('Shape train:{}\nShape test:{}'.format(x_train.shape, x_test.shape))

print("Setting parameters for Xgboost")

xgb_params = {
    'eta': 0.036,
    'max_depth': 7,
    'subsample': 0.75,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'alpha': 3.11,
    'colsample_bytree = 0.69'
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# print("Xgboost Corss validation...")
# cv_result = xgb.cv(xgb_params,
#                   dtrain,
#                   nfold=5,
#                   num_boost_round=350,
#                   early_stopping_rounds=50,
#                   verbose_eval=10,
#                   show_stdv=False
#                  )
# num_boost_rounds = len(cv_result)
# print(num_boost_rounds)
# print(cv_result)

num_boost_rounds = 282

print("\nTraining with Xgboost")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
print("\nPredicting with Xgboost")
xgb_pred2 = model.predict(dtest)

print("\nFirst Xgboost predictions:")
print(pd.DataFrame(xgb_pred2).head())


### The second XGB model
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1
}

num_boost_rounds = 239
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
xgb_pred1 = model.predict(dtest)
print( "\nSecond XGBoost predictions:" )
print( pd.DataFrame(xgb_pred1).head() )

xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
print( "\nCombined XGBoost predictions:" )
print( pd.DataFrame(xgb_pred).head() )

del train_df
del x_train
del x_test
del properties
del xgb_pred1
del xgb_pred2
del dtrain

gc.collect()

###OLS
np.random.seed(17)
random.seed(17)

train = pd.read_csv('C:/Users/J/Documents/train_2016_v2.csv', parse_dates=["transactiondate"])
properties = pd.read_csv('C:/Users/J/Documents/properties_2016.csv')
submission = pd.read_csv('C:/Users/J/Documents/sample_submission.csv')
print(len(train), len(properties), len(submission))


def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df


def MAE(y, ypred):
    # logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i] - ypred[i]) for i in range(len(y))]) / len(y)


train = pd.merge(train, properties, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
properties = []  # memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror', 'parcelid']
col = [c for c in train.columns if c not in exc]

train = get_features(train[col])
test['transactiondate'] = '2016-01-01'  # should use the most common training date
test = get_features(test[col])

reg = LinearRegression(n_jobs=-1)
reg.fit(train, y);
print('fit...')
print(MAE(y, reg.predict(train)))
train = [];
y = []  # memory

test_dates = ['2016-10-01', '2016-11-01', '2016-12-01', '2017-10-01', '2017-11-01', '2017-12-01']
test_columns = ['201610', '201611', '201612', '201710', '201711', '201712']

#### Combining two predictions results

print("\nCombining Xgboost, Lightgbm and baseline prediction...")
lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 = BASELINE_WEIGHT / (1 - OLS_WEIGHT)
pred0 = xgb_weight0 * xgb_pred + baseline_weight0 * BASELINE_PRED + lgb_weight * p_test

print("\nCombined predictions")
print(pd.DataFrame(pred0).head())

for i in range(len(test_dates)):
    test['transactiondate'] = test_dates[i]
    pred = OLS_WEIGHT * reg.predict(get_features(test)) + (1 - OLS_WEIGHT) * pred0
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

print("\nCombined XGB/LGB/baseline/OLS predictions:")
print(submission.head())

#### Write the result

print("\nPreparing to write the result")

print("\nWriting results to disk...")
submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print("Finished!")
