#%% import
from service.SDRNetwork import *
import torch
import copy
import pandas as pd
import datetime
from torch import nn
import numpy as np
#import matplotlib.pyplot as plt
import torch.nn.functional as F
from service.utils import *
import os
import pandas
#from openpyxl import load_workbook
#%% Setting config



def ask_SDRNNW_service(x_train, x_eval, x_test, y_train, y_eval, y_test,SDRNNW_params,save_file_writer,DR_results):

    max_exp_times=SDRNNW_params['max_exp_times']    #实验次数
    DR_times=SDRNNW_params['DR_times']
    verbose=SDRNNW_params['verbose']
    seed_SDR=SDRNNW_params['seed_SDR']
    NNW_params=SDRNNW_params['NNW_params']
    pure_mlp_exp_times=SDRNNW_params['pure_mlp_exp_times']

    DEVICE=x_train.device  # torch.device('cpu')
    true_p = x_train.shape[1]

    sdr_model=SDRmodel(true_p,NNW_params,DEVICE,seed=seed_SDR)



    start_time=str(datetime.datetime.now())
    # NDR
    sdr_model.train(x_train, y_train, x_eval, y_eval, max_exp_times=max_exp_times, DR_times=DR_times, verbose=verbose)

    sdrTestResult=sdr_model.test(x_test, y_test ,state="finish")
    write_yhat(sdr_model,x_train,y_train,x_eval,y_eval,x_test,y_test,save_file_writer)

    # 纯MLP的网络
    if pure_mlp_exp_times is None or pure_mlp_exp_times==0:   #是否要进行纯MLP的实验
        pure_mlp_stats=None
    else: 
        pure_mlp_stats=train_Pure_MLP(x_train, y_train, x_eval, y_eval, x_test, y_test,  true_p,  NNW_params, DEVICE,exp_times=pure_mlp_exp_times,batch_size=-1,verbose=0)

    #保存
    store_sdrNNW_result(start_time, sdr_model, sdrTestResult, pure_mlp_stats, save_file_writer)


    DR_results.append(sdr_model.get_best_DR_space)

def write_yhat(sdr_model,x_train,y_train,x_eval,y_eval,x_test,y_test,writer):
    y_train_hat=sdr_model.test(x_train,state="finish")[1]
    y_eval_hat=sdr_model.test(x_eval,state="finish")[1]
    y_test_hat=sdr_model.test(x_test,state="finish")[1]

    p=x_train.shape[1]
    columns=['x'+str(i)  for i in range(1,p+1)]+['y','yhat']
    
    df_train = pd.DataFrame( np.hstack( [x_train.detach().numpy(),y_train.detach().numpy(),y_train_hat.detach().numpy()] ) )
    df_train.columns=columns
    df_train['mode']='train'

    df_eval = pd.DataFrame( np.hstack( [x_eval.detach().numpy(),y_eval.detach().numpy(),y_eval_hat.detach().numpy()] ) )
    df_eval.columns=columns
    df_eval['mode']='eval'

    df_test = pd.DataFrame( np.hstack( [x_test.detach().numpy(),y_test.detach().numpy(),y_test_hat.detach().numpy()] ) )
    df_test.columns=columns
    df_test['mode']='test'

    df=df_train.append(df_eval).append(df_test)

    pd.DataFrame(df).to_excel(writer, sheet_name='all_data_and_prediction', index=False)
    #return x_train,y_train,y_train_hat,x_eval,y_eval,y_eval_hat,x_test,y_test,y_test_hat
    #然后拼在一起，转化为pandas，输出。


def store_sdrNNW_result(start_time,model,sdrTestResult,pure_mlp_stats,writer):
    Summarylines=[]
    Summarylines.append(f"\nmodel start at {start_time}")
    Summarylines.append(f"\nmodel end at {str(datetime.datetime.now())}")
    Summarylines.append(f"\nSDR网络使用的随机数种子: {model.seed}")
    for key,value in model.get_all_gamma_and_eye.items():
        Summarylines.append(f"r{key}: {value}")
    Summarylines.append(f"\northogonal of gammas: {str(model.check_orthogonal())} ")
    Summarylines.append(summary_test_result(sdrTestResult,pure_mlp_stats))  #测试信息


    # if os.path.exists(save_file_path):
    #     book = load_workbook(save_file_path)
    #     writer = pd.ExcelWriter(save_file_path, engine='openpyxl')
    #     writer.book = book
    #     writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    # else:
    #     writer = pd.ExcelWriter(save_file_path, engine='openpyxl')

    ## 1.save sdr stats
    pd.DataFrame(model.get_all_stats).to_excel(writer, sheet_name='sdrNNW_train_stats', index=False)
    ## 2.save summay lines
    pd.DataFrame(Summarylines).to_excel(writer, sheet_name='sdrNNW_train_summary', index=False)
    ## 3.save pure mlp stats
    pd.DataFrame(pure_mlp_stats).to_excel(writer, sheet_name='pure_mlp_stats', index=False)
    ## 4.save multi gamma stats
    pd.DataFrame(model.get_gamma_only).to_excel(writer, sheet_name='sdrNNW_multi_gamma_stats', index=False)
    ## 6.over


def summary_test_result(sdrTestResult, pure_mlp_stats):
    # Pure MLP
    ### 如果pure_mlp_stats为None（没有进行该功能），那么loss设置为-1
    pureMLP_loss=-1
    pureMLP_bestepoch=-1
    if pure_mlp_stats is not None:
        minLoss = np.inf
        bestStat = None
        for item in pure_mlp_stats:
            if item['eval_loss'] < minLoss:
                minLoss = item['eval_loss']
                bestStat = item.copy()
        pureMLP_bestepoch=bestStat['epoch']
        pureMLP_loss=bestStat['test_loss']

    # NDR
    ### 如果test error为None（没有进行该功能），那么loss设置为-1
    sdrTestLoss = sdrTestResult[2] if len(sdrTestResult) == 3 else -1

    info = f"""
    #测试：

    ##SDR网络:
    根据验证误差，选择最佳降维次数为{sdrTestResult[0]}。
    根据这个降维次数对应的参数来进行测试，NDR测试误差为{sdrTestLoss}。

    ##纯神经网络:
    选择验证误差最小的epoch{pureMLP_bestepoch}来进行测试，MLP测试误差为{pureMLP_loss}。

    """
    return info
