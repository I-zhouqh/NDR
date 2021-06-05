import pandas as pd
from config import configs
import inspect
from service.ask_data import *
from service.ask_model import ask_SDRNNW_service
#from ask_othersdr_method import *
from service.utils import Compare_estimate_and_truth, get_info_from_Generate_params,WriteData
import os
import datetime
import getpass

username = getpass.getuser()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(config):
    print("********************now start ************************")
    #print("config:",config)
    SAVEPATH=config['Global_config']['SAVEPATH']
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)
    SAVEPATH=f"{SAVEPATH}/Exp_{username}_{datetime.datetime.now().strftime('%m%d_%H%M%S')}.xlsx"
    print(SAVEPATH)

    writer = pd.ExcelWriter(SAVEPATH, engine='openpyxl')
    SDRNNW_params=config['SDRNNW_params']
    pd.DataFrame(["SDR MODEL CONFIG",str(SDRNNW_params)]).to_excel(writer, sheet_name='SDRNNW_ModelConfig', index=False)

    xsource=config['Global_config']['XSOURCE']

    Generate_params=config['Generate_params']
    x_train, x_eval, x_test, y_train, y_eval, y_test=ask_generate_data_service(Generate_params)
    true_beta=Generate_params.get('true_beta',None)
    info=get_info_from_Generate_params(Generate_params)
    pd.DataFrame(info).to_excel(writer, sheet_name='DataConfig_generate', index=False)

    WriteData(x_train,y_train,writer)
    DR_results=[]
    ask_SDRNNW_service(x_train, x_eval, x_test, y_train, y_eval, y_test,SDRNNW_params,writer,DR_results)
    #ask_othersdrmodel_service(x_train, y_train, config['Other_model_config'],DR_results)
    Compare_estimate_and_truth(DR_results,true_beta,writer)
    writer.save()
    print("********************now end ************************")


if __name__ == "__main__":
    
    for i,config in enumerate(configs):
        print(f"################# now process cofig {i+1}/{len(configs)}####################")
        main(config)