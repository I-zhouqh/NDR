import torch
import numpy as np
import pandas as pd
import random


def split_x_y(x,y,n_train,n_eval,n_test):
    
    if x.shape[0]!=y.shape[0]:
        raise ValueError
    
    x_train=x[:n_train]
    x_eval=x[n_train:(n_train+n_eval)]
    #x_test=x[(n_train+n_eval):]

    y_train=y[:n_train]
    y_eval=y[n_train:(n_train+n_eval)]
    
    #y_test=y[(n_train+n_eval):]

    # write an ugly code, to strengthen robustness
    x_test=x[-n_test:]
    y_test=y[-n_test:]
    
    return x_train,x_eval,x_test,y_train,y_eval,y_test

def generate_one_x(p,distribution_setting,custom_generate_function):
    """
    这个函数只生成一条x
    :param p:
    :param distribution_setting:
    :param custom_generate_function:
    :return: 必须是一维list（长度为p）或者一维numpy（长度为p）
    """

    if custom_generate_function is not None:
        if not callable(custom_generate_function):
            raise ValueError("custom generate function is wrongly defined !")
        return custom_generate_function()
    else:
        distribution = distribution_setting['distribution']
        params=distribution_setting['params']
        discard2=distribution_setting['discard2']
        discard3=distribution_setting['discard3']

        count=0
        while 1:  #一直循环，直到创建成功
            count+=1
            if count>=1000:
                raise ValueError("Too much cost and iteration steps for generate_one_x, please check the setting")

            if distribution=='normal':
                x = np.random.normal(params['mean'], params['sigma'], p)
            elif distribution=='uniform':
                x = np.random.uniform(params['lbound'],params['ubound'], p)
            else:
                raise ValueError("distribution %s not avaiable"%distribution)

            if discard2 is None or p==1:
                return x

            dimension_table=list(range(p))
            random.shuffle(dimension_table)

            flag_legal=True
            while len(dimension_table)>=2:
                a=dimension_table.pop()
                b=dimension_table.pop()
                if x[a]+x[b]>discard2:
                    flag_legal=False
                    break

            if flag_legal:  #如果合法，就return了，如果不合法就继续走while 1
                return x

def generate_x(n,p=None,distribution_setting=None,custom_generate_function=None):
    return np.vstack(   [generate_one_x(p,distribution_setting, custom_generate_function)  for _ in range(n)]   )



def ask_read_data_service(Readdata_params):
    # Readdata_params = {
    #                    'data_path':'',
    #                    'train_ratio': 0.7,
    #                    'eval_ratio': 0.2,
    #                    'test_ratio': 0.1,
    #                    'DEVICE': torch.device('cpu')
    #                    }
    x_centored = Readdata_params['x_centored']

    df=pd.read_csv(Readdata_params['data_path'])
    n=len(df)

    n_train=n*Readdata_params['train_ratio']
    n_eval=n*Readdata_params['eval_ratio']

    y=df['y']
    x=df.drop(['y'],axis=1)
    y=y.values
    x=x.values
    return split_x_y(x, y, n_train, n_eval,n-n_train-n_eval)


def ask_generate_data_service(Generate_params):

    n_train=Generate_params['n_train']
    n_eval=Generate_params['n_eval']
    n_test=Generate_params['n_test']
    true_p=Generate_params['true_p']
    DEVICE=Generate_params['DEVICE']
    x_distribution_setting=Generate_params['x_distribution_setting']
    seed_generate_data=Generate_params['seed_generate_data']
    true_noise_sigma=Generate_params['true_noise_sigma']
    custom_generate_function=Generate_params['custom_generate_x_function']  #生成x的函数，为None就是一些比较标准的生成正态分布、均匀分布。不为None就是用户自定义
    f=Generate_params['f']  # function for y
    x_centored=Generate_params['x_centored']
    total_n = n_train + n_eval + n_test

    if custom_generate_function is not None:
        x = generate_x(total_n, custom_generate_function= custom_generate_function)
    else:
        x=  generate_x(total_n, p=true_p, distribution_setting=x_distribution_setting)

    np.random.seed(seed_generate_data)
    random_error = np.random.normal(0, true_noise_sigma, (total_n, 1))
    y = f(x).reshape(total_n, 1) + random_error


    x = torch.tensor(x, requires_grad=False, dtype=torch.float).to(DEVICE)
    y = torch.tensor(y, requires_grad=False, dtype=torch.float).to(DEVICE)

    x_train, x_eval, x_test, y_train, y_eval, y_test = split_x_y(x, y, n_train, n_eval, n_test)

    if x_centored:    
        Mean=x_train.mean(dim=0)
        x_train-=Mean
        x_eval-=Mean
        x_test-=Mean

    return x_train, x_eval, x_test, y_train, y_eval, y_test








