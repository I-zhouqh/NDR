# coding=utf-8
import pandas as pd
import os
import re
import numpy as np

ndr_files="./data/n500"
othersdr_files="./data/OtherSDR"  #如果不使用othersdr，则写成空字符串
#othersdr_files=""
right_dimension=4


read_test_error=True
all_predictors=12



def compare_test_error(df,result):
    sentence=df[0].tolist()[-1]
    test_error_NDR,test_error_pureMLP= list(map(eval,re.findall('测试误差为(.*?)。',sentence)))

    result.append({'test_error_NDR':test_error_NDR,'test_error_pureMLP':test_error_pureMLP}  )
    
    return test_error_NDR,test_error_pureMLP

def merge_dr_results(df_ndr,df_other):
    try:
        df_other_=df_other[['method','best_k','distance','scale_of_matrix']].rename(
            columns={'best_k':'DimensionFound',
                    'distance':'space_distance_Wang',
                     'scale_of_matrix':'space_distance_MatNorm'
                    }
        )
    except:
        print("wrong input: old version of data")
        return pd.DataFrame()

    df_ndr_=df_ndr[['method','DimensionFound','space_distance_Wang','space_distance_MatNorm']]
    
    df=df_other_.append(df_ndr_)
    return df

def summary_all_df(df):
    stats=[]
    methods=df['method'].unique().tolist()
    for method in methods:
        #print(method)
        tmp=df.query("method==@method")

        stat={'method':method}
        for p in range( 1 , all_predictors):
            stat['d'+str(p)] = len( tmp.query("DimensionFound==@p") )
        stat['num']=len(tmp)
        
        stat['distance_Wang_mean']=np.mean(tmp['space_distance_Wang'])
        stat['distance_Wang_std']=np.std(tmp['space_distance_Wang'])
        stat['distance_Wang_correct_mean']=np.mean(tmp.query("DimensionFound==@right_dimension")['space_distance_Wang'])

        stat['distance_MatNorm_mean']=np.mean(tmp['space_distance_MatNorm'])
        stat['distance_MatNorm_std']=np.std(tmp['space_distance_MatNorm'])
        stat['distance_MatNorm_correct_mean']=np.mean(tmp.query("DimensionFound==@right_dimension")['space_distance_MatNorm'])
        

        #print(stat)

        stats.append(stat)
    
    return stats



test_error_list=[]
df_all=pd.DataFrame()  #承载所有结果

count=0

for file in os.listdir(ndr_files):
    if file.startswith("~"):
        continue
    if not file.endswith(".xlsx"):
        continue    

    print(file)
    
    df_ndr=pd.read_excel(ndr_files+"/"+file,sheet_name="DR_metrics_all_Method")
    
    if othersdr_files:
        df_other=pd.read_excel(othersdr_files+"/ OtherSdr "+file)
        df=merge_dr_results(df_ndr,df_other)
    else:
        df=df_ndr

    df_all=df_all.append(df)
    
    if read_test_error:
        df_test_error=pd.read_excel(ndr_files+"/"+file,sheet_name="sdrNNW_train_summary")
        compare_test_error(df_test_error,test_error_list)
    
stats=summary_all_df(df_all)



SAVEPATH='./data/summary_result'
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)

    
    
print(stats)
print(pd.DataFrame(stats))
pd.DataFrame(df_all).to_csv(f"{SAVEPATH}/raw_data.csv",index=False)
pd.DataFrame(stats).to_csv(f"{SAVEPATH}/Summary_NDR_other_DR.csv",index=False)
if read_test_error:
    pd.DataFrame(test_error_list).to_csv(f"{SAVEPATH}/Summary_NDR_pureNet.csv",index=False)
