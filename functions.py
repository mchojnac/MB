import pandas as pd
from collections import Counter
import numpy as np
import sys
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix,hstack
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.decomposition import PCA, FastICA



def ReadIn(filename,allparams):
    f=open(filename, 'r')
    for line in f.readlines():
        line=line.replace("\n","")
        if line[0]=="#":
            continue
        elif line[0]=="$":
            words=line.split(" ")
            if (words[1] in allparams.keys()):
                allparams[words[1]]=words[2:]
        else:
            words=line.split(" ")
            if len(words)==2:
                names=words[0].split("%")
                if len(names)==2:
                    if (names[0] in allparams.keys()) and (names[1] in allparams[names[0]].keys()):
                        if type(allparams[names[0]][names[1]])==str:
                            allparams[names[0]][names[1]]=str(words[1])
                        if type(allparams[names[0]][names[1]])==float:
                            allparams[names[0]][names[1]]=float(words[1])
                        if type(allparams[names[0]][names[1]])==int:
                            allparams[names[0]][names[1]]=int(words[1])
                        if type(allparams[names[0]][names[1]])==bool:
                            if words[1]=='True':
                                allparams[names[0]][names[1]]=True
                            else:
                                allparams[names[0]][names[1]]=False
    return allparams

#    file_object  = open(filename, "w")
#    for i in sorted(allparams.keys()):
#        for j in sorted(allparams[i].keys()):
#            if i=='global':
#                file_object.write("#{}%{} {}\n".format(i,j,allparams[i][j]))
#            else:
#                file_object.write("{}%{} {}\n".format(i,j,allparams[i][j]))
#    file_object.close()

def WriteSettings(filename,allparams,columns):
    file_object  = open(filename, "w")
    for i in sorted(allparams.keys()):
        if i=='columns_for_remove':
            file_object.write("$ columns_for_remove")
            for j in allparams[i]:
                file_object.write(" {}".format(j))
            file_object.write("\n")
        elif   i=='global':
            for j in sorted(allparams[i]):
                file_object.write("#{}%{} {}\n".format(i,j,allparams[i][j]))
        else:
            for j in sorted(allparams[i].keys()):
                file_object.write("{}%{} {}\n".format(i,j,allparams[i][j]))
    file_object.write("#")
    for i in columns:
        file_object.write(" {}".format(i))
    file_object.write("\n")
    file_object.close()


def Encode(df,column,maptouse):
    values=np.zeros([len(df),len(maptouse)],dtype=int)
    for idx,i in enumerate(df[column].values):
        #print(idx,i,maptouse[i],column)
        values[idx,maptouse[i]]=1
    names={"{}_{}".format(column,i) for i in range(len(maptouse))}
    print(names)
    return pd.DataFrame(values,columns=names)

def MapUsingMeans(df,name):
    mean= df[[name, 'y']].groupby([name], as_index=False).mean()
    mean_sorted=mean.sort_values('y')
    mean_sorted.reset_index(drop=True,inplace=True)
    dictformap={mean_sorted[name].values[i]:i for i in range(len(mean_sorted))}
    df[name]=df[name].map(dictformap)
    return df,dictformap

def LoadandCleanData(flagcat=True,flagcentering=0,cut=0,scaleID=0):
    df_train = pd.read_csv('train.csv')
    df_test  = pd.read_csv('test.csv')
    #total data frame
    df_train_wy=df_train.drop("y",axis=1)
    #def WriteSettings(filename,allparams,columns=None):
    df_total=pd.concat([df_train_wy,df_test])
    maxID=max(df_total['ID'].values)

    if scaleID==1:
        df_train["ID"]=df_train["ID"].apply(lambda x: x/maxID)
        df_test["ID"]=df_test["ID"].apply(lambda x: x/maxID)
    if scaleID>=2:
        df_train["ID"]=df_train["ID"].apply(lambda x: (x-0.5*maxID)/(0.5*maxID))
        df_test["ID"]=df_test["ID"].apply(lambda x: (x-0.5*maxID)/(0.5*maxID))

    catcol=list() #categorial columns
    for c,i in zip(df_train.columns,df_train.dtypes):
        if str(i).find('float')==-1 and str(i).find('int')==-1:
            catcol.append(c)

    values=list() # values for categorial columns
    for i in catcol:
        values.append(sorted(df_total[i].unique()))

    dictionaries=list() # dictonaries for categorial columns
    for i in values:
        tmpdic=dict()
        for jx,j in enumerate(i):
            tmpdic[j]=jx
        dictionaries.append(tmpdic)

    toremove=list() # columns with only one value in train do not help should be remvoe
    for i in df_train.columns:
        if len(df_train[i].unique())==1:
            toremove.append(i)

    df_train.drop(toremove,axis=1,inplace=True)
    df_test.drop(toremove,axis=1,inplace=True)
    if flagcat:
        catcol2=[]
        for i,idict in zip(catcol,dictionaries):
            df_tmp_train=Encode(df_train,i,idict)
            df_tmp_test=Encode(df_test,i,idict)
            for k in df_tmp_train.columns:
                catcol2.append(k)
            #catcol2=catcol2+df_tmp.columns
            df_train=pd.concat([df_train,df_tmp_train],axis=1)
            df_test=pd.concat([df_test,df_tmp_test],axis=1)
        df_train.drop(catcol,axis=1,inplace=True)
        df_test.drop(catcol,axis=1,inplace=True)
        df_train,df_test,toremove=RemoveDuplicats(df_train,df_test)
        for i in toremove:
            if i in catcol2:
                catcol2.remove(i)
        df_train,df_test,toremove=CleanLowVar(df_train,df_test,cut)
        for i in toremove:
            if i in catcol2:
                catcol2.remove(i)
        del catcol
        catcol=catcol2
    else:
        df_train,df_test,toremove=RemoveDuplicats(df_train,df_test)
        df_train,df_test,toremove=CleanLowVar(df_train,df_test,cut,catcol)
        for i,idict in zip(catcol,dictionaries):
            #df_train[i]=df_train[i].map(idict)
            #df_test[i]=df_test[i].map(idict)
            df_train,tmpdict=MapUsingMeans(df_train,i)
            df_test[i]=df_test[i].map(tmpdict)
            df_test[i]=df_test[i].apply(lambda x: df_train['y'].mean() if x!=float(x) else x)
            if flagcentering<1:
                continue
            l=len(idict)
            if flagcentering==1:
                df_train[i]=df_train[i].apply(lambda x: x/l)
                df_test[i]=df_test[i].apply(lambda x: x/l)
            else:
                df_train[i]=df_train[i].apply(lambda x: (x-0.5*l)/(0.5*l))
                df_test[i]=df_test[i].apply(lambda x: (x-0.5*l)/(0.5*l))

    return df_train,df_test,catcol

def RemoveDuplicats(df_train,df_test):
    toremove=list()
    for name,i in zip(df_train.columns,df_train.T.duplicated()):
        if i:
            toremove.append(name)
    #print(toremove)
    df_train.drop(toremove,axis=1,inplace=True)
    df_test.drop(toremove,axis=1,inplace=True)
    return df_train,df_test,toremove

def Encode(df,column,maptouse):
    values=np.zeros([len(df),len(maptouse)],dtype=int)
#    print(maptouse)
    for idx,i in enumerate(df[column].values):
        #print(idx,i,maptouse[i],column)
        values[idx,maptouse[i]]=1
    names={"{}_{}".format(column,i) for i in range(len(maptouse))}
    #print(names)
    return pd.DataFrame(values,columns=names)

def CleanLowVar(df_train,df_test,cut,toskip=[]):
    if cut>=1.0 or cut<=0.0:
        return df_train,df_test,[]
    if cut>0.5:
        cut=1.0-cut
    toremove=list()
    toskip.append("ID")
    toskip.append("y")
    for i,k in zip(df_train.drop(toskip,axis=1).columns,(df_train.drop(toskip,axis=1).sum()/len(df_train)).between(cut,1.0-cut)):
        if k==False:
            toremove.append(i)
    df_train.drop(toremove,axis=1,inplace=True)
    df_test.drop(toremove,axis=1,inplace=True)
    return df_train,df_test,toremove

def DoPCAICA(df_train,df_test,allparams,columns,name="rest"):
    seed=allparams['xgb']['seed']
    #if allparams['others']['RemoveID'] and "ID" in df_train.columns :
    #    df_train.drop("ID",axis=1,inplace=True)
#         df_test.drop("ID",axis=1,inplace=True)

    df_sum=pd.concat([df_train.drop("y",axis=1),df_test])
    #col=[]
    #if allparams['others']['removecatfromPCA']:
    #    col=["X0","X1","X2","X3","X4","X5","X6","X8"]
    #print(columns)
    n_comp=allparams['others']['n_comp']
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit_transform(df_sum[columns])

    pca2_results_test = pca.transform(df_test[columns])
    pca2_results_train = pca.transform(df_train[columns])

    ica = FastICA(n_components=n_comp, random_state=seed)
    ica.fit_transform(df_sum[columns])
    ica2_results_test = ica.transform(df_test[columns])
    ica2_results_train = ica.transform(df_train[columns])

    for i in range(1, n_comp+1):
        df_train['pca_'+name+ str(i)] = pca2_results_train[:,i-1]
        df_test['pca_' +name+ str(i)] = pca2_results_test[:,i-1]
        #df_train['pca2_'+name+ str(i)] = pca2_results_train[:,i-1]*pca2_results_train[:,i-1]
        #df_test['pca2_' +name+ str(i)] = pca2_results_test[:,i-1]*pca2_results_test[:,i-1]

        df_train['ica_'+name+ str(i)] = ica2_results_train[:,i-1]
        df_test['ica_'+name+ str(i)] = ica2_results_test[:, i-1]
        #df_train['ica2_'+name+ str(i)] = ica2_results_train[:,i-1]*ica2_results_train[:,i-1]
        #df_test['ica2_'+name+ str(i)] = ica2_results_test[:, i-1]*ica2_results_test[:, i-1]

    return df_train,df_test

def RemoveDuplicatsRows(df_train,df_test):
    columns=df_train.drop(["ID","y"],axis=1).columns.tolist()
    Ltrain=df_train.groupby(columns).apply(lambda x: list(x.index)).tolist()
    toremovetrain=list()
    df_train['ID'] = df_train['ID'].astype(float)
    df_test['ID'] = df_test['ID'].astype(float)
    for i in Ltrain:
        if len(i)>1:
            ym=df_train.iloc[i]['y'].mean()
            IDm=df_train.iloc[i]['ID'].mean()
            df_train['y'].values[i[0]]=ym
            df_train['ID'].values[i[0]]=IDm
            toremovetrain=toremovetrain+i[1:]
    df_train.drop(toremovetrain,inplace=True)
    df_train_wy=df_train.drop("y",axis=1)
    df_total=pd.concat([df_train_wy,df_test]).reset_index(drop=True)
    Ltotal=df_total.groupby(columns).apply(lambda x: list(x.index)).tolist()
    toremovetest=list()
    tostored=list()
    for i in Ltotal:
        if len(i)>1:
            flag=False
            valuey=0.0
            for j in i:
                if j<len(df_train):
                    flag=True
                    valuey=df_train['y'].values[j]
                if j>=len(df_train) and flag:
                    tostored.append((df_total['ID'].values[j],valuey))
                    toremovetest.append(j-len(df_train))
    df_test.drop(toremovetest,inplace=True)
    df_test.reset_index(inplace=True,drop=True)
    return df_train,df_test,tostored

def DumpMeanError(df,name,timestamp,sort=False):
    df.fillna(0,inplace=True)
    df=pd.concat([df,df.mean(axis=1).rename("mean"),df.std(axis=1).rename("std")],axis=1)
    if sort:
        df.sort_values(by="mean",ascending=False, inplace=True)
    df.to_csv('./{}_{}.csv'.format(name,timestamp),index=False)
    return df

def AddLabel(df_train,df_test):
    df_trainlabel = pd.read_csv('trainlabel.csv')
    df_testlabel  = pd.read_csv('testlabel.csv')
    valuestrain=np.zeros([len(df_trainlabel),4])
    valuestest=np.zeros([len(df_testlabel),4])
    for i in range(len(df_trainlabel)):
        valuestrain[i,df_trainlabel['label'].values[i]]=1
    for i in range(len(df_testlabel)):
        valuestest[i,df_testlabel['label'].values[i]]=1
    df_train=pd.concat([df_train,pd.DataFrame(valuestrain,columns=['l_0','l_1','l_2','l_3'])],axis=1)
    df_test=pd.concat([df_test,pd.DataFrame(valuestest,columns=['l_0','l_1','l_2','l_3'])],axis=1)
    return df_train,df_test
