import torch
import numpy as np
import datetime
from torch import nn
from scipy.stats import norm


beta1=np.array([1,1,0,0,0,0,0,0,0,0])
beta2=np.array([0,1,1,0,0,0,0,0,0,0])
beta3=np.array([0,0,1,1,0,0,0,0,0,0])
beta4=np.array([1,-1,1,-1,0,0,0,0,0,0])/2 
#beta4 = tmp/np.linalg.norm(tmp)


def generate_custom_x():
    # 生成正态分布的x1，x2，x7-x12
    n = 1
    p = 12
    sigma = np.zeros([8, 8])
    for i in range(8):
        for j in range(8):
            sigma[i][j] = 0.5 ** abs(i - j)
    res1 = np.random.multivariate_normal([0] * 8, sigma, n)
    x1 = np.array(res1[:, 0])
    x2 = np.array(res1[:, 1])

    # 生成基于绝对值的X3 X4
    ep1 = np.array(np.random.normal(0, 1, (n, 1))).reshape(n)
    ep2 = np.array(np.random.normal(0, 1, (n, 1))).reshape(n)
    x3 = abs(x1 + x2) + x1 * ep1
    x4 = abs(x1 + x2) ** 2 + x2 * ep2

    # 生成伯努利分布的X5 X6
    x5 = np.array([np.random.binomial(1, np.exp(i) / (1 + np.exp(i)), 1)[0] for i in x2])
    x6 = np.array([np.random.binomial(1, norm.cdf(i), 1)[0] for i in x2])

    # 合并所有的x
    x = np.zeros([n, 12])
    x[:, 0] = x1
    x[:, 1] = x2
    x[:, 2] = x3
    x[:, 3] = x4
    x[:, 4] = x5
    x[:, 5] = x6
    x[:, 6:12] = res1[:, 2:8]

    # 设置beta
    if p == 6:
        x = x[:, 0:6]
    elif p == 12:
        x = x
    if p != 6 and p != 12:
        raise ValueError("p %s not avaiable" % p)

    return x

# def f1(x):
#     #return x.dot(beta1.T) / (0.5 + np.power((x.dot(beta2.T)+1.5),2))
#     a1 = x.dot(beta1.T)
#     a2 = x.dot(beta2.T)
#     a3 = x.dot(beta3.T)
#     a4 = x.dot(beta4.T)
#     return a1*np.exp(0.5*a2) + a2*a3+np.sin(a3) + a4 + a4/(a1**2+1)


def f1(x):
    a1 = x.dot(beta1.T)
    a2 = x.dot(beta2.T)
    a3 = x.dot(beta3.T)
    a4 = x.dot(beta4.T)
    a5 = x.dot(beta1.T+beta3.T-beta2.T)
    return a1*np.exp(0.5*a2) + a2*a3 + np.sin(a3) + a5 + a5/(a1**2+1) + np.log(0.1+abs(a4))


# def f2(x):
#     eps = np.array(np.random.normal(0, 1, 1))
#     return np.power(x.dot(beta1.T), 2) + 2*abs(x.dot(beta2.T)) + 0.1*abs(x.dot(beta2.T))*eps
#
# def f3(x):
#     eps = np.array(np.random.normal(0, 1, 1))
#     return np.exp(x.dot(beta1.T)) + 2 * np.power((x.dot(beta2.T) + 1), 2) + abs(x.dot(beta1.T)) * eps
#
# def f4(x):
#     return np.power(x.dot(beta1.T), 2) + np.power(x.dot(beta2.T), 2)


# x_distribution_settings=[{'distribution':'normal',
#                         'params':{'mean':0,'sigma':1},
#                         'discard2':None,
#                         'discard3':None},]
                        # {'distribution':'normal',
                        # 'params':{'mean':0,'sigma':1},
                        # 'discard2':0.7,
                        # 'discard3':None},
                        # {'distribution':'normal',
                        # 'params':{'mean':0,'sigma':1},
                        # 'discard2':1.4,
                        # 'discard3':None},
                        # {'distribution':'uniform',
                        # 'params':{'lbound':-1,'ubound':1},
                        # 'discard2':None,
                        # 'discard3':None},
                        # {'distribution':'uniform',
                        # 'params':{'lbound':-1,'ubound':1},
                        # 'discard2':0.7,
                        # 'discard3':None},
                        # {'distribution':'uniform',
                        # 'params':{'lbound':-1,'ubound':1},
                        # 'discard2':1.4,
                        # 'discard3':None}]

                        
configs=[]
                        
for exp_times in range(500):

    config={
        'config_name':'_',
        'Global_config':{
                        'SAVEPATH': './data/n500' ,
                       'XSOURCE':'generate'   # must in generate, read_data 分别代表自己产生、读取现实数据
                       },

        'SDRNNW_params' : {
                        'max_exp_times': 5000,
                         'DR_times': None,
                         'verbose': 0,
                         'seed_SDR': np.random.randint(0, 100000, 1).item(),
                         'NNW_params': {'hidden': [30,30,30,30],
                                        'activation': 'relu',
                                        'loss_function': nn.MSELoss(reduction='sum'),
                                        # 'optimizer':torch.optim.Adam(self.NNWmodel.parameters(), lr=lr)
                                        'learning_rate': 3e-4,
                                        'weight_decay':1e-5
                                        },
                         'pure_mlp_exp_times':40000
                         },

        'Generate_params':{
                        'custom_generate_x_function':None, #这个参数非常关键，如果是None，就使用下面的正态分布、均匀分布之类的参数生成x。如果不是None，则用custom_generate_x_function代表的函数去自定义生成
                        'n_train':500,
                         'n_eval':300,
                         'n_test':300,
                         'true_p':10,
                         'x_distribution_setting': {'distribution':'normal',
                                                    'params':{'mean':0,'sigma':1},
                                                    'discard2':None,
                                                    'discard3':None},
                         'x_centored': True,
                         'seed_generate_data':np.random.randint(0,100000,1).item(),
                         'true_noise_sigma':0.2,
                         'f': f1,
                         'true_beta':[beta1,beta2,beta3,beta4],
                         'DEVICE': torch.device('cpu')
                        },

        'Other_model_config' :
                            {
                                'methods': ['sir','save','dr','phd','iht'],
                                'B': 500
                            }


    }



    configs.append(config)

