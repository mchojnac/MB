
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
from tqdm import tqdm
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
    others['removeoutliers']=False
    others['Ntry']=100
    others['randomsample']=True
    others['addlabel']=True
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
    Ntry=allparams['others']['Ntry']

    df_train_based,df_test_based,catcol=LoadandCleanData(allparams['others']['flagcat'],allparams['others']['flagcentering'],allparams['others']['cut'])
    #if allparams['others']['removeoutliers']:
    #    df_train_based=df_train_based[df_train_based['y']<200]
    #df_train,df_test,part=RemoveDuplicatsRows(df_train,df_test)
    idstest=df_test_based['ID'].values
    idstrain=df_train_based['ID'].values
    #y_test=list(df_train_based["y"])
    y_train_based =list (df_train_based["y"])

    if len(allparams['columns_for_remove'])>0:
        df_train_based.drop(allparams['columns_for_remove'],axis=1,inplace=True)
        df_test_based.drop(allparams['columns_for_remove'],axis=1,inplace=True)


    columnsPCA=list()
    for i in df_test_based.columns:
        if i in catcol:
            if allparams['others']['removecatfromPCA']:
                continue
        if i=="ID" and allparams['others']['RemoveID']:
            continue
        columnsPCA.append(i)


    pdimport=pd.DataFrame()
    pdlogs1=pd.DataFrame()
    pdlogs2=pd.DataFrame()
    pdypred=pd.DataFrame()
    r2list=list()
    r2testlist=list()
    loglist=list()
    logtestlist=list()
    seed=allparams['xgb']['seed']
    todrop=["y"]
    if allparams['others']['RemoveID']:
        todrop.append("ID")

    if  allparams['others']['addlabel']:
        df_train_based,df_test_based=AddLabel(df_train_based,df_test_based)

    #for i in tqdm(range(Ntry)):
    ntry=0
    while ntry<Ntry:
        allparams['xgb']['seed']=allparams['xgb']['seed']+10
        if allparams['others']['randomsample']:
            seed=seed+10
        #print(i,Ntry,allparams['xgb']['seed'])
        df_train_pca,df_test_pca=DoPCAICA(df_train_based,df_test_based,allparams,columnsPCA)
        #print(df_train_pca.columns)
        if flagtest:
            #del df_test
            frac=allparams['others']['testfraction']
            if frac<=0.0 or frac>1.0:
                allparams['others']['testfraction']=0.05
                frac=0.05
            #print(len(df_train_pca))
            df_train,df_test,y_train,y_test=train_test_split(df_train_pca,y_train_based,test_size=frac,random_state=seed)
            #print(len(df_train),len(y_train),len(df_test))
            if allparams['others']['removeoutliers']:
                df_train=df_train[df_train['y']<200]
            y_test=list(df_test["y"])
            y_train=list(df_train["y"])

            print(len(df_train),len(y_train),len(df_test))
            dtrain = xgb.DMatrix(df_train.drop(todrop, axis=1), y_train)
            dtest = xgb.DMatrix(df_test.drop(todrop, axis=1), y_test)
            #idstest=df_test['ID'].values
            #idstrain=df_train['ID'].values
            watchlist  = [(dtrain,'log'),(dtest,'test')]
        else:
            if allparams['others']['removeoutliers']:
                df_train_based=df_train_based[df_train_based['y']<200]
            y_train=y_train_based
            dtrain = xgb.DMatrix(df_train_pca.drop(todrop, axis=1), y_train_based)
            if allparams['others']['RemoveID']:
                dtest = xgb.DMatrix(df_test_pca.drop("ID", axis=1))
            else:
                dtest = xgb.DMatrix(df_test_pca)
            watchlist  = [(dtrain,'log')]

        y_mean = np.mean(y_train)
        logs=dict()
        xgb_params=allparams['xgb']
        xgb_params['base_score']=y_mean
        model = xgb.train(xgb_params, dtrain,allparams['others']['num_boost_rounds'] ,watchlist,verbose_eval=1000,evals_result=logs)

        imp=model.get_fscore()
        pdimport=pd.concat([pdimport,pd.Series(imp,name=str(i))],axis=1)
        pdlogs1=pd.concat([pdlogs1,pd.DataFrame({str(i):logs['log']['rmse']})],axis=1)
        if flagtest:
            pdlogs2=pd.concat([pdlogs2,pd.DataFrame({str(i):logs['test']['rmse']})],axis=1)
            logtestlist.append(logs['test']['rmse'][-1])
        loglist.append(logs['log']['rmse'][-1])
        if logs['log']['rmse'][-1]>7.545: #7.525
            ntry=ntry+1
        print(ntry,logs['log']['rmse'][-1])
        y_pred = model.predict(dtest)
        pdypred=pd.concat([pdypred,pd.DataFrame({str(i):y_pred})],axis=1)

    #rcParams['figure.figsize'] = 40, 40
    #xgb.plot_importance(model)
    #plt.savefig("./test/importance{}.png".format(timestamp))
    #file_imp  = open("./test/importance{}.txt".format(timestamp), "w")
        r2list.append(r2_score(dtrain.get_label(),model.predict(dtrain)))
        print("{} R2={} ".format(i,r2list[-1]))
        if flagtest:
            r2testlist.append(r2_score(dtest.get_label(),model.predict(dtest)))
            print("{} Rtest2={} ".format(i,r2testlist[-1]))
        del model
        del df_train_pca
        del df_test_pca
        if flagtest:
            del df_train
            del df_test
        del dtrain
        del dtest
    pdimport=DumpMeanError(pdimport,"./test/importance",timestamp,True)
    pdlogs1=DumpMeanError(pdlogs1,"./test/logs1",timestamp)
    if flagtest:
        pdlogs2pdlogs2=DumpMeanError(pdlogs2,"./test/logs2",timestamp)
    pdypred=DumpMeanError(pdypred,"./test/pdypred",timestamp)

    y_pred=pdypred['mean'].values

    if flagtest==False:
        y_predround=[i for i in y_pred]#+[i[1] for i in part]
        #idsfinal=[int(round(i)) for i in idstest]+[int(round(i[0])) for i in part]
        output = pd.DataFrame({'id': idstest.astype(np.int32), 'y': y_predround})
        #output = pd.DataFrame({'id': idsfinal, 'y': y_predround})
        output.to_csv('PCA{}.csv'.format(timestamp),index=False)

    rest=dict()
    rest['test']=flagtest
    allparams['global']=rest
    rest['R2']=np.mean(r2list)
    rest['R2err']=np.std(r2list)
    if flagtest:
        rest['R2testlist']=np.mean(r2testlist)
        rest['R2testerr']=np.std(r2testlist)
    WriteSettings("./test/settings{}.txt".format(timestamp),allparams,df_train_based.columns)

    fig, ax = plt.subplots()

    plt.hist(r2list,bins=np.mgrid[0.5:0.8:0.002],color='red',alpha=0.5)
    if flagtest:
        plt.hist(r2testlist,bins=np.mgrid[0.5:0.8:0.002],color='green',alpha=0.5)
    #plt.show()
    plt.savefig("./test/R2_{}.png".format(timestamp))
    plt.clf()

    if flagtest:
        #plt.hist2d(r2list,r2testlist,bins=[np.mgrid[0.5:0.8:0.002],np.mgrid[0.5:0.8:0.002]])
        #plt.colorbar()
        #plt.savefig("./test/R2D_{}.png".format(timestamp))
        #plt.hist2d(loglist,logtestlist,bins=[np.mgrid[5:9:0.2],np.mgrid[5:9:0.2]])
        plt.plot(loglist,logtestlist,"go")
        #plt.colorbar()
        plt.savefig("./test/logs2D_{}.png".format(timestamp))
        plt.clf()
    else:
        plt.plot(loglist,"go")
        #plt.colorbar()
        plt.savefig("./test/logs2D_{}.png".format(timestamp))
        plt.clf()

    #if allparams['others']['dumpresidual']:
    #    dtrain2 = xgb.DMatrix(df_train.drop('y', axis=1))
    #    ytrainpred=  model.predict(dtrain2)
    #    trainres=list()

    #    df_restrain=pd.DataFrame({'ID2':idstrain,'ypred':ytrainpred,'y':y_train})
    #    df_restest=pd.DataFrame({'ID2':idstest,'ypred':y_pred,'y':y_test})

    #    df_train=pd.concat([df_train,df_restrain],axis=1)
    #    df_test=pd.concat([df_test,df_restest],axis=1)

    #    df_train.to_csv("./res{}train.csv".format(timestamp), index = False, header = True)
    #    df_test.to_csv("./res{}test.csv".format(timestamp), index = False, header = True)
