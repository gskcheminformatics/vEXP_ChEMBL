#
# script used to investiage pred_proba threshold changes via ROC-AUC
#

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
from sklearn.metrics import roc_curve, auc
from scripts.utils import rdkit_fpconvert_numpy, rdkit_get_physchem_descr

seed = 1234

root = Path("./models")
models_list = ["ADRA2A_PIC50", "AHR_PEC50","CNR2_PIC50","HTR2C_PEC50","KCNA5_PIC50",
               "PTGS1_PIC50","PTGS2_PIC50","SLC6A2_PIC50","SLC6A4_PIC50"]

count=0
for model in models_list:
    while count < len(models_list):
        models = os.path.join(root, models_list[count])
        target = Path(models).stem
        print("This is for: "+target)
        
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
        
        # process SMILES to RDKit mols
        mols = [Chem.MolFromSmiles(smi) for smi in df.parentised_smiles]

        # generate numpy based FP with Morgan and PhysChem
        fp = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols]
        x = rdkit_fpconvert_numpy(fp)
        x = np.concatenate((x, rdkit_get_physchem_descr(mols)), axis=1)

        # train / test split using fixed seed
        x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.20, random_state=seed, stratify=y)
        
        # load and apply model scalar then models
        scale = joblib.load(str(model_sclr))
        x_tr = scale.transform(x_tr)
        x_ts = scale.transform(x_ts)
	
        xgb = joblib.load(model_xgb)
        svm = joblib.load(model_svm)
	
        pred_xgb_tr = xgb.predict(x_tr)
        pred_xgb_ts = xgb.predict(x_ts)
        pred_svm_tr = svm.predict(x_tr)
        pred_svm_ts = svm.predict(x_ts)

        # predict probabilities for the train/test set compounds
        pred_xgb_tr_prob = xgb.predict_proba(x_tr)
        pred_xgb_ts_prob = xgb.predict_proba(x_ts)
        pred_svm_tr_prob = svm.predict_proba(x_tr)
        pred_svm_ts_prob = svm.predict_proba(x_ts)

        #Save ROC-AUC Curves
        def plot_roc_curve(
            true_values, probabilities, model
        ):
            """
            Plot ROC Curve for test predictions
            """
            fig, ax = plt.subplots(1)
            
            class_order=[0,1]
            tpr=dict()
            fpr=dict()
            roc_auc=dict()
            # Plot all ROC curves
            colors = ["aqua", "darkorange", "cornflowerblue"]
            fpr, tpr, thresholds = roc_curve(true_values, probabilities[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                color=colors[0],
                label="ROC curve of {0} (area = {1:0.2f})"
                "".format(class_order[1], roc_auc),
            )
            tpr_new = tpr[1::2]
            fpr_new = fpr[1::2]
            thresholds_new = thresholds[1::2]
            count_ = 0 
            for i in tpr_new:
                j = thresholds_new[count]
                k = fpr_new[count]
                print("The TPR is: "+str(i)," and the FPR is: "+str(k),"For the following threshold: "+str(j))
                count_ += 1
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            plt.savefig(target+model+"_roc_curves.png")

        plot_roc_curve(y_ts, pred_xgb_ts_prob, "xgb")
        plot_roc_curve(y_ts, pred_svm_ts_prob, "svm")
            
        count += 1
