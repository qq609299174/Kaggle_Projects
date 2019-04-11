import pandas as pd
import numpy as np
import h5py

def save_hdf5(file_name, X,guid):
    h5f = h5py.File(file_name, 'w')
    h5f.create_dataset('X', data = X, compression="gzip", compression_opts=9)
  #  h5f.create_dataset('age', data = age, compression="gzip", compression_opts=9)
  #  h5f.create_dataset('y', data = y, compression="gzip", compression_opts=9)
    h5f.create_dataset('guid', data = guid, compression="gzip", compression_opts=9)
    h5f.close()
def add_invalid(file_name,x,y,guid):
    # Fill -1 for invalid 'days'
    df2 = pd.read_csv("~/new_transactions2.csv")
    del df2['payment_method_id']
    del df2['payment_plan_days']
    del df2['plan_list_price']
    del df2['actual_amount_paid']
    del df2['is_auto_renew']
    del df2['is_cancel']

    df2 = df2.sort_values(['userID', 'membership_expire_date'], ascending=[True, False])
    df3 = df2.drop_duplicates(subset=['userID'], keep='first')
    del df3['transaction_date']
    df3.columns = ['userID', 'end']

    df2 = df2.sort_values(['userID', 'transaction_date'], ascending=[True, True])
    df2 = df2.drop_duplicates(subset=['userID'], keep='first')
    del df2['membership_expire_date']
    df2.columns = ['userID', 'start']
    df2 = pd.merge(df2,df3,how='inner',on=['userID'])
    del df3
    print("90%")
    df2['end'] = df2['end'].astype(int)
    df2['end'] = df2['end'].apply(str)
    df2['end'] = pd.to_datetime(df2['end'])
    df2['start'] = df2['start'].astype(int)
    df2['start'] = df2['start'].apply(str)
    df2['start'] = pd.to_datetime(df2['start'])

    df2['end'] = (df2['end'] - df2['start']).dt.days
    df2['end'] = df2['end'].astype(np.int32)
    del df2['start']

    # df2['userID'] = df2['userID'].astype('S')

    # df2 = df2.groupby('userID')['end'].apply(
    #     lambda x: x.values.tolist()).to_dict()

    # Create dictionary for invalid 'days'


    # d2 = {}
    # for k, v in df2.items():
    #     flat_v = sum(v, [])[1:]
    #
    #     for idx, __ in enumerate(flat_v):
    #         flat_v[idx] += (-1) ** idx
    #     flat_v.append(789)
    #
    #     split_v = [flat_v[i:i + 2] for i in range(0, len(flat_v), 2)]
    #     d2[k] = split_v


    # del df2
    # # Insert -1 into 'x'(3D numpy array)
    # for key, value in d2.items():
    #     if key in guid:
    #         idx = np.where(guid == key)
    #         idx = idx[0][0]
    #         for i in range(len(value)):
    #             end = value[i][0]
    #             x[:,idx, end:] = -1
    for i, j in df2.iterrows():
        if j['userID'] in guid:
            idx = np.where(guid == j['userID'])
            print(idx)
            end = j['end']
            x[:, idx, end:] = -1

    x = x.astype(np.float32)
    print("99%")
    save_hdf5(file_name, x, guid)

def get_parameter(df,list_user,train):
    df = df.set_index(['userID', 'days'])
    df = df.unstack(level=-1, fill_value=0)
    df.columns = df.columns.droplevel()
    print('50%')
    df = df.reindex(columns=list(range(0, 790))).fillna(0)
    x = df.values
    print(x)
    # x = np.array(list(df.groupby('userID').apply(pd.DataFrame.as_matrix)))
    train = train.loc[train['userID'].isin(list_user)]
    #
    train = train.sort_values(['userID'], ascending=[True])
    y = train['is_churn'].values
    guid = train['userID'].values

    print('finish')
    return [x,y,guid]


print("Read the files")
df = pd.read_csv("~/new_userlogs.csv")               # Split 100000 rows
print(df.head())
df['num_25'] = df['num_25'].astype(np.int16)
df['num_50'] = df['num_50'].astype(np.int16)
df['num_75'] = df['num_75'].astype(np.int16)
df['num_985'] = df['num_985'].astype(np.int16)
df['num_100'] = df['num_100'].astype(np.int32)
df['num_unq'] = df['num_unq'].astype(np.int16)
df['date'] = df['date'].astype(np.int32)
df['userID'] = df['userID'].astype(np.int32)
transaction = pd.read_csv("~/new_transactions2.csv")         # All transaction data
transaction['transaction_date'] = transaction['transaction_date'].astype(np.int32)
transaction['userID'] = transaction['userID'].astype(np.int32)

train = pd.read_csv("~/new_test.csv")                      # All train data
train['is_churn'] = train['is_churn'].astype(np.int8)
train['userID'] = train['userID'].astype(np.int32)
print('10%')


df = pd.merge(train,df,on=['userID'],how='inner')

print('df',df.shape)
# Only keep the first transaction_date to compute 'days'
transaction = transaction.sort_values(['userID','transaction_date'], ascending=[True,True])
transaction = transaction.drop_duplicates(subset=['userID'], keep='first')

del transaction['membership_expire_date']
del transaction['payment_method_id']
del transaction['payment_plan_days']
del transaction['plan_list_price']
del transaction['actual_amount_paid']
del transaction['is_auto_renew']
del transaction['is_cancel']
transaction.columns = ['userID','start']
df = pd.merge(df,transaction,on=['userID'],how='inner')     # Merge user_log, transaction and train
print('df',df.shape)
#
print('20%')
del df['is_churn']
del transaction


df['date'] = df['date'].apply(str)
df['date'] = pd.to_datetime(df['date'])

df['start'] = df['start'].apply(str)
df['start'] = pd.to_datetime(df['start'])

df['days'] = (df['date']-df['start']).dt.days   # 'days' features by subtract between date and start
df['days'] = df['days'].astype(np.int16)
del df['date']
del df['start']
df = df[df['days'] >= 0]
df = df.sort_values(['userID'], ascending=[True])

###############################################################################################

print('39%')
print(df)
df_num_25 = df[['userID','num_25','days']]
df_num_50 = df[['userID','num_50','days']]
df_num_75 = df[['userID','num_75','days']]
df_num_985 = df[['userID','num_985','days']]
df_num_100 = df[['userID','num_100','days']]
df_num_unq = df[['userID','num_unq','days']]
df_total_secs = df[['userID','total_secs','days']]
list_num = df['userID']
del df


# #df = pd.pivot_table(df, index='userID', columns='days', fill_value=0)    # Fill 0
print('40%')
#df_num_25 = df_num_25.stack()
x1 = get_parameter(df_num_25,list_num,train)
del df_num_25
print('71%')
x2 = get_parameter(df_num_50,list_num,train)
del df_num_50
print('72%')
x3 = get_parameter(df_num_75,list_num,train)
del df_num_75
print('73%')
x4 = get_parameter(df_num_985,list_num,train)
del df_num_985
print('74%')
x5 = get_parameter(df_num_100,list_num,train)
del df_num_100
print('75%')
x6 = get_parameter(df_num_unq,list_num,train)
del df_num_unq
print('76%')
x7 = get_parameter(df_total_secs,list_num,train)
del df_total_secs
print('77%')
del train
x = np.array([x1[0],x2[0],x3[0],x4[0],x5[0],x6[0],x7[0]])
del x2
del x3
del x4
del x5
del x6
del x7
y = x1[1]
guid = x1[2]
add_invalid("kkb.h5",x,guid)
#add_invalid(name, x, y, guid)
