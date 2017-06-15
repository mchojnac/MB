
import pandas as pd
from collections import Counter
import numpy as np
import sys
import random
import functions
from scipy.sparse import csr_matrix,hstack
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from functions import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pylab import rcParams
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import r2_score
import datetime as dt
import operator

def InitSettings():
    param = dict()

    param['n_trees']= 500
    param['eta']= 0.005
    param['max_depth']= 4
    param['subsample']= 0.95
    param['objective']= 'reg:linear'
    param['eval_metric']= 'rmse'
    param['silent']= 1
    param['lambda']=1.0
    param ['alpha']=0.0 #[default=0]
    param['seed'] = 21

    others = dict()
    others['flagcat']=False
    others['flagcentering']=1
    others['cut']=0.075
    others['RemoveID']=False
    others['n_comp']=10
    others['scaleID']=0
    others['num_boost_rounds'] = 5500
    others['testfraction']=0.05
    others['removecatfromPCA']=False
    others['dumpresidual']=False
    allparams=dict()
    allparams['xgb']=param
    allparams['others']=others
    allparams['columns_for_remove']=[]
    return allparams

if __name__ == '__main__':
    timestamp=dt.datetime.now().strftime('%Y%m%d_%H_%M_%S')

    flagtest=False

    if len(sys.argv)>=3:
        filename = sys.argv[1]
        if filename=='test':
            flagtest=True
        settingsfilename =sys.argv[2]
        allparams=InitSettings()
        if settingsfilename!="default":
            allparams=ReadIn(settingsfilename,allparams)
        if len(sys.argv)>3:
            for i in sys.argv[3:]:
                par=str(i).split("=")
                if len(par)!=2:
                    continue
                parname=par[0].split("%")
                if len(parname)!=2:
                    continue
                if (parname[0] in allparams.keys()) and (parname[1] in allparams[parname[0]].keys()):
                    if type(allparams[parname[0]][parname[1]])==str:
                        allparams[parname[0]][parname[1]]=str(par[1])
                    if type(allparams[parname[0]][parname[1]])==float:
                        allparams[parname[0]][parname[1]]=float(par[1])
                    if type(allparams[parname[0]][parname[1]])==int:
                        allparams[parname[0]][parname[1]]=int(par[1])
                    if type(allparams[parname[0]][parname[1]])==bool:
                        if par[1]=='True':
                            allparams[parname[0]][parname[1]]=True
                        else:
                            allparams[parname[0]][parname[1]]=False
                elif parname[0] in allparams.keys():
                    allparams[parname[0]][parname[1]]=par[1]

    else:
        quit()



    seed=allparams['xgb']['seed']

    df_train,df_test,catcol=LoadandCleanData(allparams['others']['flagcat'],allparams['others']['flagcentering'],allparams['others']['cut'])
    idstest=df_test['ID'].values
    idstrain=df_train['ID'].values

    columnsPCA=list()
    for i in df_test.columns:
        if i in catcol:
            if allparams['others']['removecatfromPCA']:
                continue
        columnsPCA.append(i)

    df_train,df_test=DoPCAICA(df_train,df_test,allparams,columnsPCA)

    y_test=df_train["y"]
    y_train = df_train["y"]
    if len(allparams['columns_for_remove'])>0:
        y_train.drop(allparams['columns_for_remove'],axis=1,inplace=True)
        y_test.drop(allparams['columns_for_remove'],axis=1,inplace=True)

    if flagtest:
        del df_test
        frac=allparams['others']['testfraction']
        if frac<=0.0 or frac>1.0:
            allparams['others']['testfraction']=0.05
            frac=0.05
        df_train,df_test,y_train,y_test=train_test_split(df_train,y_train,test_size=frac,random_state=seed)
        dtrain = xgb.DMatrix(df_train.drop('y', axis=1), y_train)
        dtest = xgb.DMatrix(df_test.drop('y', axis=1), y_test)
        idstest=df_test['ID'].values
        idstrain=df_train['ID'].values
        watchlist  = [(dtrain,'log'),(dtest,'test')]
    else:
        dtrain = xgb.DMatrix(df_train.drop('y', axis=1), y_train)
        dtest = xgb.DMatrix(df_test)
        watchlist  = [(dtrain,'log')]

    y_mean = np.mean(y_train)

    logs=dict()
    xgb_params=allparams['xgb']
    xgb_params['base_score']=y_mean
    model = xgb.train(xgb_params, dtrain,allparams['others']['num_boost_rounds'] ,watchlist,verbose_eval=50,evals_result=logs)


    rcParams['figure.figsize'] = 40, 40
    xgb.plot_importance(model)
    plt.savefig("./test/importance{}.png".format(timestamp))
    file_imp  = open("./test/importance{}.txt".format(timestamp), "w")
    imp=model.get_fscore()
    sorted_x = sorted(imp.items(), key=operator.itemgetter(1), reverse=True)
    for i in sorted_x:
        file_imp.write("{}={}\n".format(i[0],i[1]))
    file_imp.close()

    y_pred = model.predict(dtest)
    if flagtest==False:
        output = pd.DataFrame({'id': idstest.astype(np.int32), 'y': y_pred})
        output.to_csv('PCA{}.csv'.format(timestamp),index=False)

    rest=dict()
    rest['test']=flagtest
    allparams['global']=rest
    rest['R2']=r2_score(dtrain.get_label(),model.predict(dtrain))
    if flagtest:
        rest['R2test']=r2_score(dtest.get_label(),model.predict(dtest))
    WriteSettings("./test/settings{}.txt".format(timestamp),allparams,df_train.columns)
    if flagtest:
        out=pd.DataFrame({'train':logs['log']['rmse'],"test":logs['test']['rmse']})
    else:
        out=pd.DataFrame({'train':logs['log']['rmse']})
    out.to_csv("./test/logs{}.csv".format(timestamp), index = False, header = True)

    if allparams['others']['dumpresidual']:
        dtrain2 = xgb.DMatrix(df_train.drop('y', axis=1))
        ytrainpred=  model.predict(dtrain2)
        trainres=list()

        df_restrain=pd.DataFrame({'ID2':idstrain,'ypred':ytrainpred,'y':y_train})
        df_restest=pd.DataFrame({'ID2':idstest,'ypred':y_pred,'y':y_test})

        df_train=pd.concat([df_train,df_restrain],axis=1)
        df_test=pd.concat([df_test,df_restest],axis=1)

        df_train.to_csv("./res{}train.csv".format(timestamp), index = False, header = True)
        df_test.to_csv("./res{}test.csv".format(timestamp), index = False, header = True)
