import pandas as pd
from collections import Counter
import numpy as np
import sys
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix,hstack
from sklearn.preprocessing import LabelEncoder,OneHotEncoder




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

#def WriteSettings(filename,allparams,columns=None):
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


def LoadandCleanData(flagcat=True,flagcentering=0,cut=0,scaleID=0):
    df_train = pd.read_csv('train.csv')
    df_test  = pd.read_csv('test.csv')
    #total data frame
    df_train_wy=df_train.drop("y",axis=1)
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
        for i,idict in zip(catcol,dictionaries):
            df_train=pd.concat([df_train,Encode(df_train,i,idict)],axis=1)
            df_test=pd.concat([df_test,Encode(df_test,i,idict)],axis=1)
        df_train.drop(catcol,axis=1,inplace=True)
        df_test.drop(catcol,axis=1,inplace=True)
        df_train,df_test=RemoveDuplicats(df_train,df_test)
        df_train,df_test=CleanLowVar(df_train,df_test,cut)
    else:
        df_train,df_test=RemoveDuplicats(df_train,df_test)
        df_train,df_test=CleanLowVar(df_train,df_test,cut,catcol)
        for i,idict in zip(catcol,dictionaries):
            df_train[i]=df_train[i].map(idict)
            df_test[i]=df_test[i].map(idict)
            if flagcentering<1:
                continue
            l=len(idict)
            if flagcentering==1:
                df_train[i]=df_train[i].apply(lambda x: x/l)
                df_test[i]=df_test[i].apply(lambda x: x/l)
            else:
                df_train[i]=df_train[i].apply(lambda x: (x-0.5*l)/(0.5*l))
                df_test[i]=df_test[i].apply(lambda x: (x-0.5*l)/(0.5*l))

    return df_train,df_test

def RemoveDuplicats(df_train,df_test):
    toremove=list()
    for name,i in zip(df_train.columns,df_train.T.duplicated()):
        if i:
            toremove.append(name)
    #print(toremove)
    df_train.drop(toremove,axis=1,inplace=True)
    df_test.drop(toremove,axis=1,inplace=True)
    return df_train,df_test

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
        return df_train,df_test
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
    return df_train,df_test
