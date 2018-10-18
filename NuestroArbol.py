
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import numpy.ma as ma
import math
from collections import Counter


# In[2]:



# In[3]:


# Definición de la estructura del árbol. 

class Hoja:
    #  Contiene las cuentas para cada clase (en forma de diccionario)
    #  Por ejemplo, {'Si': 2, 'No': 2}
    def __init__(self, etiquetas):
        self.cuentas = dict(Counter(etiquetas.compressed()))


class Nodo_De_Decision:
    # Un Nodo de Decisión contiene preguntas y una referencia al sub-árbol izquierdo y al sub-árbol derecho
     
    def __init__(self, pregunta, sub_arbol_izquierdo, sub_arbol_derecho):
        self.pregunta = pregunta
        self.sub_arbol_izquierdo = sub_arbol_izquierdo
        self.sub_arbol_derecho = sub_arbol_derecho
        
        
# Definición de la clase "Pregunta"
class Pregunta:
    def __init__(self, atributo, valor):
        self.atributo = atributo
        self.valor = valor
    
    def cumple(self, instancia):
        # Devuelve verdadero si la instancia cumple con la pregunta
        return instancia[self.atributo] >= self.valor[0] and instancia[self.atributo] < self.valor[1]
    
    def __repr__(self):
        return "¿Es el valor para {} entre {} y {}?".format(self.atributo, self.valor[0], self.valor[1])


# In[4]:
        



def partir_segun(pregunta, instancias, etiquetas):
    # Esta función debe separar instancias y etiquetas según si cada instancia cumple o no con la pregunta (ver método 'cumple')
    # COMPLETAR (recomendamos utilizar máscaras para este punto) 
    mascaraCumplen = np.apply_along_axis(pregunta.cumple, 1, instancias)
    
    
    instancias_cumplen = instancias[np.array(mascaraCumplen)]
    instancias_no_cumplen = instancias[np.array(~mascaraCumplen)]
    if isinstance(etiquetas, np.ma.core.MaskedArray):
        etiquetas_cumplen = ma.array(etiquetas.compressed(), mask = ~mascaraCumplen);
        etiquetas_no_cumplen = ma.array(etiquetas.compressed(), mask = mascaraCumplen);
    else:
        etiquetas_cumplen = ma.array(etiquetas, mask = ~mascaraCumplen);
        etiquetas_no_cumplen = ma.array(etiquetas, mask = mascaraCumplen);
    
    
    return instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen


# In[5]:


def imprimir_arbol(arbol, spacing=""):
    if isinstance(arbol, Hoja):
        print (spacing + "Hoja:", arbol.cuentas)
        return

    print (spacing + str(arbol.pregunta))

    print (spacing + '--> True:')
    imprimir_arbol(arbol.sub_arbol_izquierdo, spacing + "  ")

    print (spacing + '--> False:')
    imprimir_arbol(arbol.sub_arbol_derecho, spacing + "  ")



# In[ ]:

def buildColumnValues(X):
    result = {}
    n = X.shape[1]
    for i in np.arange(n):
        col = np.sort(X[0:,i]) 
        j = 0
        result[i] = []
        while j + 99 < len(col) - 1:
            result[i].append((col[j],col[j+99]))
            j = j + 100
    return result

def predecir(arbol, x_t):
    if isinstance(arbol, Hoja):
       return max(arbol.cuentas, key=arbol.cuentas.get)
        
    else:
        if arbol.pregunta.cumple(x_t):
            return predecir(arbol.sub_arbol_izquierdo, x_t);
        else:
            return predecir(arbol.sub_arbol_derecho, x_t);
        
   
def predictProbabilities(arbol, x_t):
    if isinstance(arbol, Hoja):
        total = sum(arbol.cuentas.values());
        if 1 in arbol.cuentas:
           probabilityEqualTo1 = arbol.cuentas[1]/total;
           return np.array([1-probabilityEqualTo1,probabilityEqualTo1]);
        else:
            return np.array([1,0]);
        
    else:
        if arbol.pregunta.cumple(x_t):
            return predictProbabilities(arbol.sub_arbol_izquierdo, x_t);
        else:
            return predictProbabilities(arbol.sub_arbol_derecho, x_t);     
        
        
class MiClasificadorArbol(): 
    def __init__(self, max_depth, criterion):
        self.arbol = None
        self.altura_maxima = max_depth
        self.criterio = criterion
    
    def fit(self, X_train, y_train):
        self.columnas = np.arange(X_train.shape[1])
        self.columnValues = buildColumnValues(X_train.values);
        self.etiquetasSet = set(y_train);
        self.arbol = self.construir_arbol(X_train.values, y_train, self.altura_maxima, self.criterio)
        return self
    
    def predictValue(self, X_test):
        predictions = []
        for x_t in X_test.values:
            prediction = predecir(self.arbol, x_t)
            predictions.append(prediction)
        return predictions
    
    def predict_proba(self, X_test):
        predictions = []
        for x_t in X_test.values:
            prediction = predictProbabilities(self.arbol, x_t)
            predictions.append(prediction)
        return np.array(predictions)    
    
    def score(self, X, y):
        y_pred = self.predictValue(X)
        
        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y)) / len(y)
        return accuracy
    
    def construir_arbol(self, instancias, etiquetas, altura_maxima, criterio):
        # ALGORITMO RECURSIVO para construcción de un árbol de decisión binario. 
        
        # Suponemos que estamos parados en la raiz del árbol y tenemos que decidir cómo construirlo. 
        ganancia, pregunta = self.encontrar_mejor_atributo_y_corte(instancias, etiquetas, criterio)
        
        # Criterio de corte: ¿Hay ganancia?
        if ganancia == 0 or altura_maxima == 0:
            #  Si no hay ganancia en separar, no separamos. 
            return Hoja(etiquetas)
        else: 
            # Si hay ganancia en partir el conjunto en 2
            instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = partir_segun(pregunta, instancias, etiquetas)
            # partir devuelve instancias y etiquetas que caen en cada rama (izquierda y derecha)
            
            if isinstance(altura_maxima, int):
                proxima_altura_maxima = altura_maxima - 1
            else:
                proxima_altura_maxima = None
    
            # Paso recursivo (consultar con el computador más cercano)
            sub_arbol_izquierdo = self.construir_arbol(instancias_cumplen, etiquetas_cumplen, proxima_altura_maxima, criterio)
            sub_arbol_derecho   = self.construir_arbol(instancias_no_cumplen, etiquetas_no_cumplen, proxima_altura_maxima, criterio)
            # los pasos anteriores crean todo lo que necesitemos de sub-árbol izquierdo y sub-árbol derecho
            
            # sólo falta conectarlos con un nodo de decisión:
            return Nodo_De_Decision(pregunta, sub_arbol_izquierdo, sub_arbol_derecho)
        
    def encontrar_mejor_atributo_y_corte(self, instancias, etiquetas, criterio):
        max_ganancia = 0
        mejor_pregunta = None
        for columna in self.columnas:
            for valor in self.columnValues[columna]:
                # Probando corte para atributo y valor
                pregunta = Pregunta(columna, valor)
                _, etiquetas_rama_izquierda, _, etiquetas_rama_derecha = partir_segun(pregunta, instancias, etiquetas)
                
                if(criterio == 'entropy'):
                    ganancia = self.ganancia_informacion(etiquetas, etiquetas_rama_izquierda, etiquetas_rama_derecha)
                else:
                    ganancia = self.ganancia_gini(etiquetas, etiquetas_rama_izquierda, etiquetas_rama_derecha)
                
                if ganancia > max_ganancia:
                    max_ganancia = ganancia
                    mejor_pregunta = pregunta
                
        return max_ganancia, mejor_pregunta
    
    def entropia(self, etiquetas):
        if isinstance(etiquetas, np.ma.core.MaskedArray):
            dicc = dict(Counter(etiquetas.compressed()));
        else:
            dicc =  dict(Counter(etiquetas));
        totalEtiquetas = sum(dicc.values())
        result = 0
        for etiqueta in dicc.keys():
            proporcion = dicc[etiqueta]/totalEtiquetas;
            result = result - proporcion*(math.log(proporcion,2));
        return result
    
    
    def ganancia_informacion(self, etiquetas, etiquetas_rama_izquierda, etiquetas_rama_derecha):
#        print("rama izquierda")
#        print(etiquetas_rama_izquierda)
#        print("rama derecha")
#        print(etiquetas_rama_derecha)
        totalIzquierda = etiquetas_rama_izquierda.count()
        totalDerecha =  etiquetas_rama_derecha.count()                                  
        totalEtiquetas = totalIzquierda + totalDerecha;
        entropiaIzquierdaPonderada = (totalIzquierda/totalEtiquetas)*self.entropia(etiquetas_rama_izquierda);
        entropiaDerechaPonderada = (totalDerecha/totalEtiquetas)*self.entropia(etiquetas_rama_derecha);
        entropiaInicial = self.entropia(etiquetas)
        return entropiaInicial - entropiaIzquierdaPonderada - entropiaDerechaPonderada;
    
    def gini(self, etiquetas):
        if isinstance(etiquetas, np.ma.core.MaskedArray):
            dicc = dict(Counter(etiquetas.compressed()));
        else:
            dicc =  dict(Counter(etiquetas));
            
        totalEtiquetas = sum(dicc.values())
        impureza = 1
        for etiqueta in dicc.keys():
            impureza = impureza - (dicc[etiqueta]/totalEtiquetas)**2
        return impureza
    
    def ganancia_gini(self, etiquetas, etiquetas_rama_izquierda, etiquetas_rama_derecha):
        totalIzquierda = etiquetas_rama_izquierda.count()
        totalDerecha =  etiquetas_rama_derecha.count()                                 
        totalEtiquetas = totalIzquierda + totalDerecha;
        giniIzquierdaPonderado = (totalIzquierda/totalEtiquetas)*self.gini(etiquetas_rama_izquierda);
        giniDerechaPonderado = (totalDerecha/totalEtiquetas)*self.gini(etiquetas_rama_derecha);
        giniInical = self.gini(etiquetas);
        return giniInical - giniIzquierdaPonderado - giniDerechaPonderado;    


