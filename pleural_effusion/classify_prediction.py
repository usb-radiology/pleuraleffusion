# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Load libraries
import pandas as pd
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
import json
from functools import reduce
import pickle
import os
import argparse


source=os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-f", dest="feature", required=True)
parser.add_argument("-frad", dest="frad", required=True)
parser.add_argument("-meta", dest="meta", required=True)
parser.add_argument("-model_dir", dest="model_dir", default="{0}/../ML_models".format(source))

args = parser.parse_args()

f = open (args.frad, "r")
frad = json.load(f)
frad = {key: value for key, value in frad[0].items() if key.startswith("original_")}

if bool(frad):

    f = open (args.feature, "r")
    fpred = json.load(f)
    features = fpred.copy()
    features.update(frad)

    classes = ["blood","easy","enhance","gas","lobulated"]

    if os.path.exists(args.meta):
        meta = json.load(open (args.meta))
    else:
        meta = {}
        
    meta["classes"]={}
    for c in range(len(classes)):
        json_features = os.path.join(args.model_dir,"ML_features_{0}_label.json".format(classes[c]))
        modelpath = os.path.join(args.model_dir,"ML_model_{0}_label.sav".format(classes[c]))
        f = open (json_features, "r")
        fsel = json.load(f)

        X_feature = np.zeros(len(fsel))
        for f in range(len(fsel)):
            X_feature[f] = features[fsel[f]]

        X_feature = np.squeeze(X_feature)

        loaded_model = pickle.load(open(modelpath, 'rb'))
        result = loaded_model.predict_proba([X_feature])
        meta["classes"][classes[c]]=int(result[0][1]*100)

    json.dump(meta, open(args.meta, 'w'))



# %%



