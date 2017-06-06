
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

    df_train,df_test=LoadandCleanData(allparams['others']['flagcat'],allparams['others']['flagcentering'],allparams['others']['cut'])

    if allparams['others']['RemoveID']:
        df_train.drop("ID",axis=1,inplace=True)
        df_test.drop("ID",axis=1,inplace=True)

    df_sum=pd.concat([df_train.drop("y",axis=1),df_test])

    n_comp=allparams['others']['n_comp']
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit_transform(df_sum)
    pca2_results_test = pca.transform(df_test)
    pca2_results_train = pca.transform(df_train.drop(["y"], axis=1))

    ica = FastICA(n_components=n_comp, random_state=seed)
    ica.fit_transform(df_sum)
    ica2_results_test = ica.transform(df_test)
    ica2_results_train = ica.transform(df_train.drop(["y"], axis=1))


    for i in range(1, n_comp+1):
        df_train['pca_' + str(i)] = pca2_results_train[:,i-1]
        df_test['pca_' + str(i)] = pca2_results_test[:,i-1]

        df_train['ica_' + str(i)] = ica2_results_train[:,i-1]
        df_test['ica_' + str(i)] = ica2_results_test[:, i-1]




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
        watchlist  = [(dtrain,'log'),(dtest,'test')]
    else:
        dtrain = xgb.DMatrix(df_train.drop('y', axis=1), y_train)
        dtest = xgb.DMatrix(df_test)
        watchlist  = [(dtrain,'log')]
    ids=df_test['ID'].values
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
    output = pd.DataFrame({'id': ids.astype(np.int32), 'y': y_pred})
    output.to_csv('PCA{}.csv'.format(timestamp),index=False)

    rest=dict()
    rest['test']=flagtest
    allparams['global']=rest
    rest['R2']=r2_score(model.predict(dtrain), dtrain.get_label())
    if flagtest:
        rest['R2test']=r2_score(model.predict(dtest), dtest.get_label())
    WriteSettings("./test/settings{}.txt".format(timestamp),allparams,df_train.columns)
    if flagtest:
        out=pd.DataFrame({'train':logs['log']['rmse'],"test":logs['test']['rmse']})
    else:
        out=pd.DataFrame({'train':logs['log']['rmse']})
    out.to_csv("./test/logs{}.csv".format(timestamp), index = False, header = True)
