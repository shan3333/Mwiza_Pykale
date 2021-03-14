"""
Preprocessing of ABIDE datasets: 
- functions for extracting phenotype features: 
Reference: https://github.com/kundaMwiza/fMRI-site-adaptation/blob/97b0d26a4bf199efe82c2e1d52fcb93986eb10da/imports/preprocess_data.py
"""

import os
import numpy as np
from scipy import sio
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

def preprocess_phenotypes(pheno_ft, params):
    """
    Preprocess phenotypes. 
    Categorical -> ordinal representation
    """

    if params['model'] == 'MIDA':
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2])], remainder ='passthrough')
    else:
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder ='passthrough')

    pheno_ft = ct.fit_transform(pheno_ft)
    pheno_ft = pheno_ft.astype('float32')

    return(pheno_ft)

def phenotype_ft_vector(pheno_ft, num_subjects, model='MIDA'):
    """
    Create phenotype feature vector to concatenate with fmri feature vectors
    Args:
        pheno_ft: 
        num_subjects: number of subjects
        model (str): defaults to 'MIDA'
            Options: MIDA or raw
    Return:

    """
    gender = pheno_ft[:,0]
    if model == 'MIDA':
        eye = pheno_ft[:, 0]
        hand = pheno_ft[:,2]
        age = pheno_ft[:,3]
        fiq = pheno_ft[:,4]
    else:
        eye = pheno_ft[:, 2]
        hand = pheno_ft[:,3]
        age = pheno_ft[:,4]
        fiq = pheno_ft[:,5]

    phenotype_ft = np.zeros((num_subjects,4))
    phenotype_ft_eye = np.zeros((num_subjects, 2))
    phenotype_ft_hand = np.zeros((num_subjects, 3))
        
    for i in range(num_subjects):
        phenotype_ft[i, int(gender[i])] = 1
        phenotype_ft[i, -2] = age[i]
        phenotype_ft[i, -1] = fiq[i]
        phenotype_ft_eye[i, int(eye[i])] = 1
        phenotype_ft_hand[i, int(hand[i])] = 1
    
    if model == 'MIDA':
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand], axis=1)
    else:
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand, phenotype_ft_eye], axis=1)

    return phenotype_ft

def get_subject_score(subject_list, score):
    """
    Get phenotype values for a list of subjects
    Args: 
        subject_list : list of subject IDs
    Return:

    """
    scores_dict = {}
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                if score == 'HANDEDNESS_CATEGORY':
                    if (row[score].strip() == '-9999') or (row[score].strip() ==''):
                        scores_dict[row['SUB_ID']] = 'R'
                    elif row[score] == 'Mixed':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    elif row[score] == 'L->R':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    else:
                        scores_dict[row['SUB_ID']] = row[score]
                elif (score == 'FIQ' or score == 'PIQ' or score == 'VIQ'):
                    if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                        scores_dict[row['SUB_ID']] = 100
                    else:
                        scores_dict[row['SUB_ID']] = float(row[score])

                else:
                    scores_dict[row['SUB_ID']] = row[score]

    return scores_dict

# Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, subject_list):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs
    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    pheno_ft = pd.DataFrame()
    global_phenos = []

    for i, l in enumerate(scores):
        phenos = []
        label_dict = get_subject_score(subject_list, l)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                phenos.append(float(label_dict[subject_list[k]]))
        else:
            for k in range(num_nodes):
                phenos.append(label_dict[subject_list[k]])
        global_phenos.append(phenos)
    
    for i, l in enumerate(scores):
        pheno_ft.insert(i, l, global_phenos[i], True)

    return pheno_ft 