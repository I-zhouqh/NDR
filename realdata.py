import pandas as pd
import inspect
from service.ask_data import *
from service.ask_model import ask_SDRNNW_service
#from ask_othersdr_method import *
from service.utils import Compare_estimate_and_truth, get_info_from_Generate_params,WriteData
import os
import datetime
import getpass
from torch import nn
from sklearn import datasets


username = getpass.getuser()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SDRNNW_params=   {
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
                    }

x_centored=True
DEVICE=torch.device('cpu')
SAVEPATH='./data/bostan'
FILENAME='bostan_NDR5'
n_train=400
n_eval=106
n_test=106

# we should specify the columns for x and y
y_columns=None
x_columns=None  # 'x'如果是None的话，就是除了y列的所有列。y不能是None


print("********************now start ************************")
#print("config:",config)


if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)
SAVEPATH=f"{SAVEPATH}/{FILENAME}.xlsx"
print(SAVEPATH)


writer = pd.ExcelWriter(SAVEPATH, engine='openpyxl')

pd.DataFrame(["SDR MODEL CONFIG",str(SDRNNW_params)]).to_excel(writer, sheet_name='SDRNNW_ModelConfig', index=False)

bostan = datasets.load_boston()
X = bostan.data
n=X.shape[0]
drop_col=bostan.feature_names.tolist().index('CHAS')  # chas is binary, drop it
X = np.delete(X,drop_col,axis=1)

y = bostan.target
y = y.reshape(len(y), 1)


X = torch.tensor(X, requires_grad=False, dtype=torch.float).to(DEVICE)
y = torch.tensor(y, requires_grad=False, dtype=torch.float).to(DEVICE)

x_train, x_eval, x_test, y_train, y_eval, y_test = split_x_y(X, y, n_train, n_eval, n_test)

print(X.shape)
print(x_train.shape,y_train.shape)
print(x_eval.shape,y_eval.shape)
print(x_test.shape,y_test.shape)


if x_centored:    
    Mean=x_train.mean(dim=0)
    x_train-=Mean
    x_eval-=Mean
    x_test-=Mean
    
WriteData(x_train,y_train,writer)
DR_results=[]
ask_SDRNNW_service(x_train, x_eval, x_test, y_train, y_eval, y_test,SDRNNW_params,writer,DR_results)
#ask_othersdrmodel_service(x_train, y_train, config['Other_model_config'],DR_results)
#Compare_estimate_and_truth(DR_results,true_beta,writer)
writer.save()
print("********************now end ************************")
