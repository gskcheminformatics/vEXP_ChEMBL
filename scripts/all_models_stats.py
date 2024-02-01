#
# script to:
# read the directory list
# for each model dir read data, scalar and model
# run train/test predictions then generate metrics
#

import glob
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
from scripts.utils import rdkit_fpconvert_numpy, rdkit_get_physchem_descr

# this seed should be fixed the same as the seed used to build the model
seed = 1234

root = Path("./models/")
models = glob.glob(str(root / "*C50"))

for model in models:

	target = Path(model).stem
        
	#print(f"Working on {target} in dir {root}")
	model_tsv = root / target / str(target + "_train.tsv")
	model_ecc = root / target / str(target + "_ecc.pkl")
	model_brf = root / target / str(target + "_brf.pkl")
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
	
	brf = joblib.load(model_brf)
	ecc = joblib.load(model_ecc)
	
	# predict for train (tr) and test (ts)
	pred_brf_tr = brf.predict(x_tr)
	pred_brf_ts = brf.predict(x_ts)
	pred_ecc_tr = ecc.predict(x_tr)
	pred_ecc_ts = ecc.predict(x_ts)

	# calc statistics: {model}_{metric}_{test/train}
	brf_prc_tr = precision_score(y_tr, pred_ecc_tr)
	brf_rcl_tr = recall_score(y_tr, pred_ecc_tr)
	brf_roc_tr = roc_auc_score(y_tr, pred_ecc_tr)
	brf_bac_tr = balanced_accuracy_score(y_tr, pred_brf_tr)
	brf_mcc_tr = matthews_corrcoef(y_tr, pred_brf_tr)
	brf_kpp_tr = cohen_kappa_score(y_tr, pred_brf_tr)
	
	brf_prc_ts = precision_score(y_ts, pred_ecc_ts)
	brf_rcl_ts = recall_score(y_ts, pred_ecc_ts)
	brf_roc_ts = roc_auc_score(y_ts, pred_ecc_ts)
	brf_bac_ts = balanced_accuracy_score(y_ts, pred_brf_ts)
	brf_mcc_ts = matthews_corrcoef(y_ts, pred_brf_ts)
	brf_kpp_ts = cohen_kappa_score(y_ts, pred_brf_ts)
	
	ecc_prc_tr = precision_score(y_tr, pred_ecc_tr)
	ecc_rcl_tr = recall_score(y_tr, pred_ecc_tr)
	ecc_roc_tr = roc_auc_score(y_tr, pred_ecc_tr)
	ecc_bac_tr = balanced_accuracy_score(y_tr, pred_ecc_tr)
	ecc_mcc_tr = matthews_corrcoef(y_tr, pred_ecc_tr)
	ecc_kpp_tr = cohen_kappa_score(y_tr, pred_ecc_tr)
	
	ecc_prc_ts = precision_score(y_ts, pred_ecc_ts)
	ecc_rcl_ts = recall_score(y_ts, pred_ecc_ts)
	ecc_roc_ts = roc_auc_score(y_ts, pred_ecc_ts)
	ecc_bac_ts = balanced_accuracy_score(y_ts, pred_ecc_ts)
	ecc_mcc_ts = matthews_corrcoef(y_ts, pred_ecc_ts)
	ecc_kpp_ts = cohen_kappa_score(y_ts, pred_ecc_ts)
	
	pstr = f"{target},"
	pstr += f"{ecc_prc_tr},{ecc_rcl_tr},{ecc_roc_tr},"
	pstr += f"{ecc_bac_tr},{ecc_mcc_tr},{ecc_kpp_tr},"
	pstr += f"{ecc_prc_ts},{ecc_rcl_ts},{ecc_roc_ts},"
	pstr += f"{ecc_bac_ts},{ecc_mcc_ts},{ecc_kpp_ts},"
	pstr += f"{brf_prc_tr},{brf_rcl_tr},{brf_roc_tr},"
	pstr += f"{brf_bac_tr},{brf_mcc_tr},{brf_kpp_tr},"
	pstr += f"{brf_prc_ts},{brf_rcl_ts},{brf_roc_ts},"
	pstr += f"{brf_bac_ts},{brf_mcc_ts},{brf_kpp_ts}"

	print(pstr)





