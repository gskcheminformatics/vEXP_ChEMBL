#
# utils used across notebooks and scripts
#
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Descriptors


def rdkit_fpconvert_numpy(fp):
    output = []
    for f in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)

def rdkit_get_physchem_descr(mols):
    descr = []
    for m in mols:
        descr.append([Descriptors.MolLogP(m),
                        Descriptors.TPSA(m),
                        Descriptors.NumHAcceptors(m),
                        Descriptors.NumHDonors(m),
                        Descriptors.NumRotatableBonds(m),
                        Descriptors.FractionCSP3(m)])
    return np.asarray(descr)
