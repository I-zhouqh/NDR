import torch
import pandas as pd
from torch import nn
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import inspect

class MLP(nn.Module):
    def __init__(self, dimension_list,activation='relu'):
        super(MLP, self).__init__()

        if not isinstance(dimension_list,list) or len(dimension_list)<2:
            raise ValueError("Check the layer num of NNW!")

        self.linears=nn.ModuleList()

        for i in range(len(dimension_list)-1):
            self.linears.append( nn.Linear(dimension_list[i],dimension_list[i+1])  )

        if activation=='relu':
            self.activation=nn.LeakyReLU()
        elif activation=='sigmoid':
            self.activation=nn.Sigmoid()
        elif activation=='tanh':
            self.activation = nn.Tanh()

    def forward(self, x):

        for i,layer in enumerate(self.linears):
            x = layer(x)
            if i!=len(self.linears)-1:#最后一层就不激活了
                x = self.activation(x)
        return x   # 返回的是没有softmax的东西

def WriteData(x,y,writer):
    xdata=x.detach().numpy()
    ydata=y.detach().numpy().reshape(-1,1)

    p=xdata.shape[1]
    colnames  =[ "x"+str(i)  for i in range(1,p+1) ]
    colnames.append("y")

    data=np.hstack([xdata,ydata])

    df=pd.DataFrame(data)
    df.columns=colnames
    df.to_excel(writer, sheet_name='Data', index=False)


def draw_plot(all_stats, param_names):
    # 根据统计信息画图
    all_stats = pd.DataFrame(all_stats)
    n = len(all_stats)
    x = all_stats['epoch']

    # draw param of NNW
    for param in param_names:
        y1 = all_stats[param + '.mean']
        y2 = all_stats[param + '.std']
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, 'r', label="mean")
        ax1.legend(loc=1)
        ax1.set_ylabel('mean')
        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(x, y2, 'g', label="std")
        ax2.legend(loc=2)
        ax2.set_xlim([1, n])
        ax2.set_ylabel('std')
        # ax2.set_xlabel('Same X for both exp(-x) and ln(x)');
        plt.title(param)
        plt.show()

    # draw gamma
    cols_with_gamma=[item for item in list(all_stats.columns) if  item.startswith('r_')]  #含有gamma的那些列名
    for col in cols_with_gamma:
        plt.plot(x, all_stats[col])
        plt.title('gamma %s'%col)
        plt.show()

    # draw loss
    plt.plot(x, all_stats['train_loss'])

    if all_stats['eval_loss'][0] is not None:  # 把test也加上
        plt.plot(x, all_stats['eval_loss'])
    plt.legend()
    plt.title('loss')
    plt.show()


#def train_Pure_MLP(X_train,y_train,true_p,hidden,exp_times,loss_function,device,lr=1e-5,x_eval=None,y_eval=None,verbose=0, batch_size=-1 ):
def train_Pure_MLP(x_train, y_train, x_eval, y_eval, x_test, y_test,  true_p,  NNW_params, DEVICE,exp_times=50000,batch_size=-1,verbose=0):
    print("### TRAIN Pure MLP STARTS ###")

    hidden=NNW_params['hidden']
    loss_function=NNW_params['loss_function']

    if not isinstance(hidden,list):
        hidden=[hidden]

    PureMLP = MLP([true_p]+ hidden + [1])
    #PureMLP.apply(weights_init_uniform_rule)
    PureMLP.to(DEVICE)
    optimizer = torch.optim.Adam(PureMLP.parameters(), lr=NNW_params['learning_rate'])

    train_data=MyDataset(x_train,y_train)

    batch_size=batch_size if batch_size!=-1 else len(train_data)  #如果batch_size=-1，那就不分batch
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    stats=[]

    for i in range(1, exp_times):
        total_train_loss=0
        for step, batch_data in enumerate(train_dataloader):
            x_batch,y_batch=batch_data

            yhat = PureMLP(x_batch)

            if yhat.shape!=y_batch.shape:
                raise ValueError("dimension not match!!")
            loss = loss_function(yhat, y_batch)
            total_train_loss+=loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # optimizer只会更新网络中的参数，不会更新r

        eval_loss=None
        if x_eval is not None and y_eval is not None:
            y_eval_hat=PureMLP(x_eval)
            eval_loss = loss_function(y_eval, y_eval_hat)

        test_loss=None
        if x_test is not None and y_test is not None:
            y_test_hat=PureMLP(x_test)
            test_loss = loss_function(y_test, y_test_hat)

        stats.append(  {
                        'epoch':i,
                        'train_loss':total_train_loss.item(),
                        'eval_loss':eval_loss.item(),
                        'test_loss': test_loss.item()
                        }          )

    print("### TRAIN Pure MLP ENDS ###")

    return stats


class MyDataset(Dataset):

    def __init__(self, x, y):
        self.X=x
        self.y=y

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()  # in case that the index is a tensor type
        return self.X[index,:],self.y[index]

    def __len__(self):
        return len(self.X)


def get_info_from_Generate_params(Generate_params):
    f_x = Generate_params['custom_generate_x_function']
    f_y = Generate_params['f']

    info = ["generate config", str(Generate_params), "generate x code"]
    info += [inspect.getsource(f_x) if f_x is not None else " "]
    info += ["generate y code", inspect.getsource(f_y)]

    return info

def Compare_estimate_and_truth(DR_results,true_beta,writer):
    if true_beta is None or len(DR_results)==0:
        return

    true_beta=np.vstack(true_beta).T
    true_beta = np.matrix(true_beta)
    true_space=true_beta.dot((true_beta.T.dot(true_beta)).I).dot(true_beta.T)

    for item in DR_results:
        # item has key method, DRspace, dimension
        estimate_space=item['EstimatedSpace']
        k=true_space-estimate_space
        uh, singular_value, vh = np.linalg.svd(k)   #奇异值分解，而不是特征值分解

        item['true_basis']=true_beta
        item['true_space']=true_space
        item['space_distance_Wang']=np.nanmax(singular_value).item()
        item['space_distance_MatNorm']=np.linalg.norm(k,ord=None).item()   #None means Frobenius norm

    pd.DataFrame(DR_results).to_excel(writer, sheet_name='DR_metrics_all_Method', index=False)
    pd.DataFrame(true_space).to_excel(writer, sheet_name='true_space',index=False)