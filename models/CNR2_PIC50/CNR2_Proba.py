import os 
import numpy as np
import pandas as pd
import joblib
import xgboost
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score

seed = 1234

root = Path("./models")
model = ["CNR2_PIC50"]

models = os.path.join(root,model[0])
target = Path(models).stem
        
#print(f"Working on {target} in dir {root}")
model_tsv = root / target / str(target + "_train.tsv")
model_svm = root / target / str(target + "_svm.pkl")
model_xgb = root / target / str(target + "_xgb.pkl")
model_sclr = root / target / str("scalar_" + target + ".pkl")
        
# setup data
df = pd.read_csv(str(model_tsv), sep="\t")
        
y_str = df.class_label.tolist() # convert str to list
y_int = pd.get_dummies(y_str) # one hot encode the str labels
y_new = y_int["POSITIVE"] # set new y to one hot encoded POSITIVE label column 
df['class_label_binary'] = y_new # set new encoded label into dataframe
y = df.class_label_binary.to_list()
        
# process SMILES to features
mols = [Chem.MolFromSmiles(smi) for smi in df.parentised_smiles]

# generate binary Morgan fingerprint with radius 2
fp = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols]
# convert to numpy array

def rdkit_numpy_convert(fp):
    output = []
    for f in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)

x = rdkit_numpy_convert(fp)

# add physchem descriptors
descr = []
for m in mols:
    descr.append([Descriptors.MolLogP(m),
                Descriptors.TPSA(m),
                Descriptors.NumHAcceptors(m),
                Descriptors.NumHDonors(m),
                Descriptors.NumRotatableBonds(m),
                Descriptors.FractionCSP3(m)])
descr = np.asarray(descr)
# add to morgan FP's
x = np.concatenate((x, descr), axis=1)

# split the data using same seed as build
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.20, random_state=seed, stratify=y)
        
# load and apply scalar
scale = joblib.load(str(model_sclr))
x_tr = scale.transform(x_tr)
x_ts = scale.transform(x_ts)
        
# load model model 
xgb = joblib.load(model_xgb)  # <--- if you uncomment this line it fails
svm = joblib.load(model_svm)
        
#Test Set Validation
# predict for the train/test set compounds
pred_xgb_tr = xgb.predict(x_tr)
pred_xgb_ts = xgb.predict(x_ts)
pred_svm_tr = svm.predict(x_tr)
pred_svm_ts = svm.predict(x_ts)

# predict probabilities for the train/test set compounds --> XGB FIRST

threshold = 0.84
pred_xgb_tr_prob = xgb.predict_proba(x_tr)
pred_xgb_ts_prob = xgb.predict_proba(x_ts)

# Set probabilities as column 2 --> XGB FIRST
new_proba_xgb = pred_xgb_ts_prob[:, 1]

# Set new label as 1 if greater than threshold of new predict proba value --> XGB FIRST
new_prediction_xgb = new_proba_xgb > threshold

# Now do same for SVM with different threshold
threshold = 0.77
pred_svm_tr_prob = svm.predict_proba(x_tr)
pred_svm_ts_prob = svm.predict_proba(x_ts)
new_proba_svm = pred_svm_ts_prob[:, 1]
new_prediction_svm = new_proba_svm > threshold

# Calculate old statistics
	
xgb_prc_ts = precision_score(y_ts, pred_svm_ts)
xgb_rcl_ts = recall_score(y_ts, pred_svm_ts)
xgb_roc_ts = roc_auc_score(y_ts, pred_svm_ts)
xgb_bac_ts = balanced_accuracy_score(y_ts, pred_xgb_ts)
xgb_mcc_ts = matthews_corrcoef(y_ts, pred_xgb_ts)
xgb_kpp_ts = cohen_kappa_score(y_ts, pred_xgb_ts)
	
svm_prc_ts = precision_score(y_ts, pred_svm_ts)
svm_rcl_ts = recall_score(y_ts, pred_svm_ts)
svm_roc_ts = roc_auc_score(y_ts, pred_svm_ts)
svm_bac_ts = balanced_accuracy_score(y_ts, pred_svm_ts)
svm_mcc_ts = matthews_corrcoef(y_ts, pred_svm_ts)
svm_kpp_ts = cohen_kappa_score(y_ts, pred_svm_ts)

old = f"{target},"
old += f"{svm_prc_ts},{svm_rcl_ts},{svm_roc_ts},"
old += f"{svm_bac_ts},{svm_mcc_ts},{svm_kpp_ts},"
old += f"{xgb_prc_ts},{xgb_rcl_ts},{xgb_roc_ts},"
old += f"{xgb_bac_ts},{xgb_mcc_ts},{xgb_kpp_ts},"

print(old)

# Calculate new statistics

xgb_prc_ts = precision_score(y_ts, new_prediction_xgb)
xgb_rcl_ts = recall_score(y_ts, new_prediction_xgb)
xgb_roc_ts = roc_auc_score(y_ts, new_prediction_xgb)
xgb_bac_ts = balanced_accuracy_score(y_ts, new_prediction_xgb)
xgb_mcc_ts = matthews_corrcoef(y_ts, new_prediction_xgb)
xgb_kpp_ts = cohen_kappa_score(y_ts, new_prediction_xgb)
        
svm_prc_ts = precision_score(y_ts, new_prediction_svm)
svm_rcl_ts = recall_score(y_ts, new_prediction_svm)
svm_roc_ts = roc_auc_score(y_ts, new_prediction_svm)
svm_bac_ts = balanced_accuracy_score(y_ts, new_prediction_svm)
svm_mcc_ts = matthews_corrcoef(y_ts, new_prediction_svm)
svm_kpp_ts = cohen_kappa_score(y_ts, new_prediction_svm)
        
pstr = f"{target},"
pstr += f"{svm_prc_ts},{svm_rcl_ts},{svm_roc_ts},"
pstr += f"{svm_bac_ts},{svm_mcc_ts},{svm_kpp_ts},"
pstr += f"{xgb_prc_ts},{xgb_rcl_ts},{xgb_roc_ts},"
pstr += f"{xgb_bac_ts},{xgb_mcc_ts},{xgb_kpp_ts},"

print(pstr)
        