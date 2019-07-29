#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Draw, DataStructs
from rdkit.Chem import inchi

import os

#import pubchempy
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import confusion_matrix
from itertools import product
import itertools
import seaborn as sns


# In[ ]:


# read the smiles format
X_train = pd.read_csv('chem_train.csv',skiprows=[0],names=['smiles'])
X = X_train.drop(columns=['smiles'])
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('chem_test.csv',skiprows=[0],names=['smiles'])
X_sp = X_test.drop(columns=['smiles'])
X_test['mol'] = X_test['smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 
y_test=pd.read_csv('y_test.csv')
#X_train
#y_train
#X_test


# # Descriptors
# 
# rdkit.Chem.Descriptors provides a number of general molecular descriptors that can also be used to featurize a molecule. Most of the descriptors are straightforward to use from Python.
# 
# Using this package we can add some useful features to our model:
# 
# rdkit.Chem.Descriptors.TPSA() - the surface sum over all polar atoms or molecules also including their attached hydrogen atoms;
# rdkit.Chem.Descriptors.ExactMolWt() - exact molecural weight;
# rdkit.Chem.Descriptors.NumValenceElectrons() - number of valence electrons (may illustrate general electronic density)
# rdkit.Chem.Descriptors.NumHeteroatoms() - general number of non-carbon atoms.
# rdkit.Chem.Descriptors.FpDensityMorgan1(x)
# rdkit.Chem.Descriptors.FpDensityMorgan2(x)
# rdkit.Chem.Descriptors.FpDensityMorgan3(x)

# In[5]:


from rdkit.Chem import Descriptors


X_train['mol'] = X_train['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

X_train['tpsa'] = X_train['mol'].apply(lambda x: Descriptors.TPSA(x))
X_train['mol_w'] = X_train['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
X_train['num_valence_electrons'] = X_train['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
X_train['num_heteroatoms'] = X_train['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
X_train['morgan_fingerprints1'] = X_train['mol'].apply(lambda x: Descriptors.FpDensityMorgan1(x))
X_train['morgan_fingerprints2'] = X_train['mol'].apply(lambda x: Descriptors.FpDensityMorgan2(x))
X_train['morgan_fingerprints3'] = X_train['mol'].apply(lambda x: Descriptors.FpDensityMorgan3(x))
#X_train['mf'] = X_train['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048))


# In[6]:


X_test['mol'] =X_test['smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 

#Extract descriptors
X_test['tpsa'] = X_test['mol'].apply(lambda x: Descriptors.TPSA(x))
X_test['mol_w'] = X_test['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
X_test['num_valence_electrons'] = X_test['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
X_test['num_heteroatoms'] = X_test['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
X_test['morgen_fingerprints1'] = X_test['mol'].apply(lambda x: Descriptors.FpDensityMorgan1(x))
X_test['morgen_fingerprints2'] = X_test['mol'].apply(lambda x: Descriptors.FpDensityMorgan2(x))
X_test['morgen_fingerprints3'] = X_test['mol'].apply(lambda x: Descriptors.FpDensityMorgan3(x))
#X_test['mf'] = X_test['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048))


# In[7]:


from mol2vec.mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
model = word2vec.Word2Vec.load('model_300dim.pkl')
#Constructing sentences
X_train['sentence'] = X_train.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

#Extracting embeddings to a numpy.array

#Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
X_train['mol2vec'] = [DfVec(x) for x in sentences2vec(X_train['sentence'], model, unseen='UNK')]
X_mol = np.array([x.vec for x in X_train['mol2vec']])
X_mol = pd.DataFrame(X_mol)

#Concatenating matrices of features
X_train_new = pd.concat((X, X_mol), axis=1)


# In[8]:


#perform mol2vec for the X_test as well 
from mol2vec.mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
model = word2vec.Word2Vec.load('model_300dim.pkl')
#Constructing sentences
X_test['sentence'] = X_test.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

#Extracting embeddings to a numpy.array
#Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
X_test['mol2vec'] = [DfVec(x) for x in sentences2vec(X_test['sentence'], model, unseen='UNK')]
X_mol_test = np.array([x.vec for x in X_test['mol2vec']])
X_mol_test = pd.DataFrame(X_mol_test)

#Concatenating matrices of features
X_test_new = pd.concat((X_sp, X_mol_test), axis=1)


# In[ ]:


import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=4)
kf.get_n_splits(X_train_new)

print(kf)  

for train_index, test_index in kf.split(X_train_new):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_train_new[train_index], X_train_new[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[59]:


from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

#mods=doModel()
#cv_results = [cross_validate(m, X_train_new, y_train, cv=4) for m in mods]
#sorted(cv_results[0].keys())                         

#cv_results['test_score'] 


# # Parameters:
# 
# hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
# The ith element represents the number of neurons in the ith hidden layer.
# 
# alpha : float, optional, default 0.0001
# L2 penalty (regularization term) parameter.
# 
# max_iter : int, optional, default 200
# Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.
# 
# momentum : float, default 0.9
# Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.
# 
# solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
# The solver for weight optimization.
# 
# ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
# ‘sgd’ refers to stochastic gradient descent.
# ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

# In[9]:


#multilayer percepton

from sklearn.neural_network import MLPClassifier
models = [] 
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=0.0001, 
                     solver='sgd', verbose=10,  random_state=10, momentum = 0.7, tol=0.000000001)


def doModel2():
    for i in range (0,9):
        task = 'Task'+str(i+1)
        y = y_train[task].values
        clf.fit(X_train_new, y) 
        models.append(clf)
    return models


# # Parameters
# 
# min_samples_leaf : int, float, optional (default=1)
# The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
# 
# If int, then consider min_samples_leaf as the minimum number.
# If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

# In[48]:


from sklearn.ensemble import RandomForestClassifier
models = [] 
clf = RandomForestClassifier(n_estimators=50, n_jobs=8, random_state=0, verbose=1, max_depth=None,
                                  min_samples_leaf=4)


def doModel():
    for i in range (0,9):
        task = 'Task'+str(i+1)
        y = y_train[task].values
        clf.fit(X_train_new, y) 
        models.append(clf)
    return models


# In[10]:


print(y_test.shape)
predictions = np.zeros(y_test.shape)
i = 0
models = doModel2()
print(len(models))
for model in models:
    task = 'Task'+str(i+1)
    x_train = X_train_new[y_train[task] !=-1]
    y_Train = y_train[task][y_train[task]!=-1]
    model.fit(x_train,y_Train)
    e = model.predict(X_test_new)
    predictions[:,i]=e
    i+=1
np.savetxt("y_test_m4.csv", predictions, delimiter=",")
print(predictions)


# In[3]:





# In[ ]:




