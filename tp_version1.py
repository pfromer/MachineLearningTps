#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import scipy as sp
from scipy.misc import comb
from sklearn.model_selection import train_test_split , KFold, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pdb

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importo datos 

# In[111]:


X_comp = pd.read_csv('X_competencia.csv')
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')
X.drop(['index'],inplace=True, axis=1)
y.drop(['index'], inplace=True,axis=1)
X.isnull().any().any()


# In[74]:


X_desarrollo , X_holout ,y_desarrollo, y_holdout = train_test_split(X, y['output'],
                                                                    test_size=0.13,random_state=0,stratify=y['output'])
y_desarrollo_np=np.array(y_desarrollo)


# ## Arbolito

# In[104]:


#esto esta al pedo lo puse para recordar
arbol = DecisionTreeClassifier(max_depth=3, criterion="gini")
arbol.fit(X_desarrollo, y_desarrollo)


# ### K-fold CV 

# In[112]:


a=[]
kfold = KFold(n_splits=5)
kfold.get_n_splits(X_desarrollo)
#este for itera sobre los k folds en cada loop tego un set de datos y otro de validacion
for train, test  in kfold.split(X_desarrollo):
    #print("TRAIN:", train_index,'\n', "TEST:", test_index,'\n' )
    X_train, X_val = X_desarrollo.loc[train], X_desarrollo.loc[test]
    y_train, y_val = y_desarrollo_np[train], y_desarrollo_np[test]
    #intancio el arbol que voy a entrenar en cada fold
    arbol = DecisionTreeClassifier(max_depth=2, criterion="gini")
    #pdb.set_trace()        
    #estoy tratando de sacar los NaN que aparecen en los datos al partir, no se porque aparecen pero el fit no funca sino
    if  X_train.isnull().any().any()==True:
        X_train.dropna(axis='columns',inplace=True)
        
    arbol.fit(X_train, y_train.astype(int))
    a=append(arbol.score(X=X_test, y=y_test))
    pdb.set_trace()
print(a)    
    


# In[ ]:





# In[ ]:


var_comp = x_comp.var(axis=0)
var_X= X.var(axis=0)
#np.column_stack((g,h,g-h))


# In[ ]:


np.sort(np.abs(g-h))


# In[ ]:


g.shape


# In[ ]:


Df = pd.DataFrame({'var1':var_comp,'var2':var_X,'rest' :var_comp-var_X})
Df


# In[ ]:


Df.sort_values(by=['var1'])


# In[ ]:


Df.sort_values(by=['var2'])

