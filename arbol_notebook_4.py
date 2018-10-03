
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import numpy.ma as ma
from collections import Counter


# In[2]:


def construir_arbol(instancias, etiquetas):
    # ALGORITMO RECURSIVO para construcción de un árbol de decisión binario. 
    
    # Suponemos que estamos parados en la raiz del árbol y tenemos que decidir cómo construirlo. 
    ganancia, pregunta = encontrar_mejor_atributo_y_corte(instancias, etiquetas)
    
    # Criterio de corte: ¿Hay ganancia?
    if ganancia == 0:
        #  Si no hay ganancia en separar, no separamos. 
        return Hoja(etiquetas)
    else: 
        # Si hay ganancia en partir el conjunto en 2
        instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = partir_segun(pregunta, instancias, etiquetas)
        # partir devuelve instancias y etiquetas que caen en cada rama (izquierda y derecha)

        # Paso recursivo (consultar con el computador más cercano)
        sub_arbol_izquierdo = construir_arbol(instancias_cumplen, etiquetas_cumplen)
        sub_arbol_derecho   = construir_arbol(instancias_no_cumplen, etiquetas_no_cumplen)
        # los pasos anteriores crean todo lo que necesitemos de sub-árbol izquierdo y sub-árbol derecho
        
        # sólo falta conectarlos con un nodo de decisión:
        return Nodo_De_Decision(pregunta, sub_arbol_izquierdo, sub_arbol_derecho)


# In[3]:


# Definición de la estructura del árbol. 

class Hoja:
    #  Contiene las cuentas para cada clase (en forma de diccionario)
    #  Por ejemplo, {'Si': 2, 'No': 2}
    def __init__(self, etiquetas):
        self.cuentas = dict(Counter(etiquetas))


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
        return instancia[self.atributo] == self.valor
    
    def __repr__(self):
        return "¿Es el valor para {} igual a {}?".format(self.atributo, self.valor)


# In[4]:


def gini(etiquetas):    
    dicc =  dict(Counter(etiquetas));
    totalEtiquetas = sum(dicc.values())
    impureza = 1
    for etiqueta in dicc.keys():
        impureza = impureza - (dicc[etiqueta]/totalEtiquetas)**2
    return impureza

def ganancia_gini(etiquetas, etiquetas_rama_izquierda, etiquetas_rama_derecha):
    giniInicial = gini(etiquetas)
    totalIzquierda = len(etiquetas_rama_izquierda);
    totalDerecha =  len(etiquetas_rama_derecha)                                  
    totalEtiquetas = totalIzquierda + totalDerecha;
    giniIzquierdaPonderado = (totalIzquierda/totalEtiquetas)*gini(etiquetas_rama_izquierda);
    giniDerechaPonderado = (totalDerecha/totalEtiquetas)*gini(etiquetas_rama_derecha);
    giniInical = gini(etiquetas);
    ganancia_gini = giniInical - giniIzquierdaPonderado - giniDerechaPonderado;
    return ganancia_gini


def partir_segun(pregunta, instancias, etiquetas):
    # Esta función debe separar instancias y etiquetas según si cada instancia cumple o no con la pregunta (ver método 'cumple')
    # COMPLETAR (recomendamos utilizar máscaras para este punto) 
    mascaraCumplen = instancias.dropna().apply(lambda x: pregunta.cumple(x), axis=1)
    instancias_cumplen = instancias.dropna().mask(-mascaraCumplen);
    instancias_no_cumplen = instancias.dropna().mask(mascaraCumplen);
    etiquetas_cumplen = ma.array(etiquetas, mask = -mascaraCumplen).compressed();
    etiquetas_no_cumplen = ma.array(etiquetas, mask = mascaraCumplen).compressed();
    
    return instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen


# In[5]:


def encontrar_mejor_atributo_y_corte(instancias, etiquetas):
    max_ganancia = 0
    mejor_pregunta = None
    for columna in instancias.columns:
        for valor in set(instancias[columna]):
            # Probando corte para atributo y valor
            pregunta = Pregunta(columna, valor)
            _, etiquetas_rama_izquierda, _, etiquetas_rama_derecha = partir_segun(pregunta, instancias, etiquetas)
   
            ganancia = ganancia_gini(etiquetas, etiquetas_rama_izquierda, etiquetas_rama_derecha)
            
            if ganancia > max_ganancia:
                max_ganancia = ganancia
                mejor_pregunta = pregunta
            
    return max_ganancia, mejor_pregunta


def imprimir_arbol(arbol, spacing=""):
    if isinstance(arbol, Hoja):
        print (spacing + "Hoja:", arbol.cuentas)
        return

    print (spacing + str(arbol.pregunta))

    print (spacing + '--> True:')
    imprimir_arbol(arbol.sub_arbol_izquierdo, spacing + "  ")

    print (spacing + '--> False:')
    imprimir_arbol(arbol.sub_arbol_derecho, spacing + "  ")


# In[6]:


X = pd.DataFrame([["Sol","Calor","Alta","Debil"],
                ["Sol","Calor","Alta","Fuerte"],
                ["Nublado","Calor","Alta","Debil"],
                ["Lluvia","Templado","Alta","Debil"],
                ["Lluvia","Frio","Normal","Debil"],
                ["Lluvia","Frio","Normal","Fuerte"],
                ["Nublado","Frio","Normal","Fuerte"],
                ["Sol","Templado","Alta","Debil"],
                ["Sol","Frio","Normal","Debil"],
                ["Lluvia","Templado","Normal","Debil"],
                ["Sol","Templado","Normal","Fuerte"],
                ["Nublado","Templado","Alta","Fuerte"],
                ["Nublado","Calor","Normal","Debil"],
                ["Lluvia","Templado","Alta","Fuerte"]],
                columns = ['Cielo', 'Temperatura', 'Humedad', 'Viento'])

y = ['No', 'No', 'Si', 'Si', 'Si', 'No', 'Si', 'No', 'Si', 'Si', 'Si', 'Si', 'Si', 'No']

display(X)
display(y)


# In[7]:


arbol = construir_arbol(X, y)
imprimir_arbol(arbol)


# ## Resultado esperado
# 
# ```
# ¿Es el valor para Cielo igual a Nublado?
# --> True:
#   ¿Es el valor para Temperatura igual a Frio?
#   --> True:
#     Hoja: {'Si': 1}
#   --> False:
#     ¿Es el valor para Temperatura igual a Templado?
#     --> True:
#       Hoja: {'Si': 1}
#     --> False:
#       Hoja: {'Si': 2}
# --> False:
#   ¿Es el valor para Humedad igual a Normal?
#   --> True:
#     ¿Es el valor para Viento igual a Fuerte?
#     --> True:
#       Hoja: {'No': 1, 'Si': 1}
#     --> False:
#       ¿Es el valor para Cielo igual a Sol?
#       --> True:
#         Hoja: {'Si': 1}
#       --> False:
#         Hoja: {'Si': 2}
#   --> False:
#     ¿Es el valor para Cielo igual a Sol?
#     --> True:
#       ¿Es el valor para Temperatura igual a Templado?
#       --> True:
#         Hoja: {'No': 1}
#       --> False:
#         Hoja: {'No': 2}
#     --> False:
#       Hoja: {'Si': 1, 'No': 1}
# ```

# ## Parte 2 (opcional)
# Protocolo sklearn para clasificadores. Completar el protocolo requerido por sklearn. Deben completar la función predict utilizando el árbol para predecir valores de nuevas instancias. 
# 

# In[8]:


def predecir(arbol, x_t):
    if isinstance(arbol, Hoja):
       return max(arbol.cuentas, key=arbol.cuentas.get)
        
    else:
        if arbol.pregunta.cumple(x_t):
            return predecir(arbol.sub_arbol_izquierdo, x_t);
        else:
            return predecir(arbol.sub_arbol_derecho, x_t);
        
    return "Si"
        
class MiClasificadorArbol(): 
    def __init__(self):
        self.arbol = None
        self.columnas = ['Cielo', 'Temperatura', 'Humedad', 'Viento']
    
    def fit(self, X_train, y_train):
        self.arbol = construir_arbol(pd.DataFrame(X_train, columns=self.columnas), y_train)
        return self
    
    def predict(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t], columns=self.columnas).iloc[0]
            prediction = predecir(self.arbol, x_t_df) 
            print(x_t, "predicción ->", prediction)
            predictions.append(prediction)
        return predictions
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y_test)) / len(y_test)
        return accuracy
        

# Ejemplo de uso
clf = MiClasificadorArbol()

# Tomar en cuenta que sklearn espera numpy arrays:
clf.fit(np.array(X), y)
clf.score(np.array(X), y)

