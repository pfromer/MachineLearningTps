
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import scipy as sp
from scipy.misc import comb
from sklearn.model_selection import train_test_split , KFold, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pdb
from sklearn.metrics import roc_auc_score
import arbol_clasificador as ac

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importo datos 

# In[ ]:


X_comp = pd.read_csv('X_competencia.csv')
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')
X.drop(['index'],inplace=True, axis=1)
y.drop(['index'], inplace=True,axis=1)


# In[ ]:


X_desarrollo , X_holout ,y_desarrollo, y_holdout = train_test_split(X, y['output'],
                                                                    test_size=0.13,random_state=0,stratify=y['output'])
y_desarrollo_np=np.array(y_desarrollo)


# ### K-fold CV 

# In[ ]:


acc_train=[]
acc_val=[]
ROC_AUC_train=[]
ROC_AUC_val=[]


kfold = StratifiedKFold(n_splits=5)
kfold.get_n_splits(X_desarrollo,y_desarrollo)
#este for itera sobre los k folds en cada loop tego un set de datos y otro de validacion
for train, test  in kfold.split(X_desarrollo,y_desarrollo):
    #print("TRAIN:", train_index,'\n', "TEST:", test_index,'\n' )
    X_train, X_val = X_desarrollo.iloc[train], X_desarrollo.iloc[test]
    y_train, y_val = y_desarrollo_np[train], y_desarrollo_np[test]
    #intancio el arbol que voy a entrenar en cada fold
    arbol = ac.MiClasificadorArbol(max_depth=3, criterion="gini")
    
    arbol.fit(X_train, y_train.astype(int))
    acc_train.append(arbol.score(X=X_train, y=y_train))
    acc_val.append(arbol.score(X=X_val, y=y_val))
    ROC_AUC_train.append(roc_auc_score(y_train,arbol.predict(X_train)))
    ROC_AUC_val.append(roc_auc_score(y_val,arbol.predict(X_val)))





# # Tabla de precision

# In[ ]:


tabla_prec = pd.DataFrame({ 'Partición' : np.arange(1,6),'Accuracy(training)' :acc_train,
                          'Accuracy(validation)' : acc_val,
                          'ROC AUC(train)' : ROC_AUC_train,
                          'ROC AUC(val)' : ROC_AUC_val})
display(tabla_prec)  


# # Arboles combinaciones

# In[6]:


def accu(alturas,criterio):
    acc_train_2=[]
    acc_val_2=[]



    kfold = StratifiedKFold(n_splits=5)
    kfold.get_n_splits(X_desarrollo,y_desarrollo)
    #este for itera sobre los k folds en cada loop tego un set de datos y otro de validacion
    for train, test  in kfold.split(X_desarrollo,y_desarrollo):
        #print("TRAIN:", train_index,'\n', "TEST:", test_index,'\n' )
        X_train, X_val = X_desarrollo.iloc[train], X_desarrollo.iloc[test]
        y_train, y_val = y_desarrollo_np[train], y_desarrollo_np[test]
        #intancio el arbol que voy a entrenar en cada fold
        arbol = ac.MiClasificadorArbol(max_depth=3, criterion="gini")

        arbol.fit(X_train, y_train.astype(int))
        acc_train_2.append(arbol.score(X=X_train, y=y_train))
        acc_val_2.append(arbol.score(X=X_val, y=y_val))
    return (np.mean(acc_train_2),np.mean(acc_val_2))

        





#acc_train_2=[]
#acc_val_2=[]
#altura=[]
#alt={3:'3',5:'5', None:'Infinito'}
#alturas = [3,5,None]
#corte = []
#for alturas in alturas:
#    criterio =['gini','entropy']
#    for criterio in criterio:
#        acc_train_2.append(accu(alturas,criterio)[0])
#        
#        acc_val_2.append(accu(alturas,criterio)[1])
#        altura.append(alt[alturas])
#        corte.append(criterio)
#
#
# # Tabla con combinaciones
#
#
#
#
#tabla_comb = pd.DataFrame({ 'Altura Máxima' : altura,
#                          'Criterio de evaluación de corte' : corte,
#                          'Accuracy(training)' : acc_train_2,
#                          'Accuracy(validation)' : acc_val_2})
#tabla_comb.sort_values(by=['Criterio de evaluación de corte'],ascending=False)
#
#display(tabla_comb)  