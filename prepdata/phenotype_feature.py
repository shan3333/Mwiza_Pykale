"""
Preprocessing of ABIDE datasets: 
- functions for extracting phenotype features: 
Reference: https://github.com/kundaMwiza/fMRI-site-adaptation/blob/97b0d26a4bf199efe82c2e1d52fcb93986eb10da/imports/preprocess_data.py
"""

import os
import csv
import numpy as np
from scipy import sio
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

def preprocess_phenotypes(pheno_ft_cat, model):
    """
    Preprocess phenotypes. 
    Args:
        pheno_ft_cat: phenotype categorical representation
    Return:
        pheno_ft_ord: phenotype ordinal representation
    """

    if model == 'MIDA':
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2])], remainder ='passthrough')
    else:
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder ='passthrough')

    pheno_ft_ord = ct.fit_transform(pheno_ft_cat)
    pheno_ft_ord = pheno_ft_ord.astype('float32')

    return(pheno_ft_ord)

def phenotype_ft_vector(pheno_ft_ord, num_subjects, model='MIDA'):
    """
    Create phenotype feature vector to concatenate with fmri feature vectors
    Args:
        pheno_ft_ord: ordinal representation
        num_subjects: number of subjects
        model (str): defaults to 'MIDA', options: MIDA or raw
    Return:
        phenotype_ft_vec: phenotype feature vector to concatenate with fmri feature vectors
    """
    gender = pheno_ft_ord[:,0]
    if model == 'MIDA':
        eye = pheno_ft_ord[:, 0]
        hand = pheno_ft_ord[:,2]
        age = pheno_ft_ord[:,3]
        fiq = pheno_ft_ord[:,4]
    else:
        eye = pheno_ft_ord[:, 2]
        hand = pheno_ft_ord[:,3]
        age = pheno_ft_ord[:,4]
        fiq = pheno_ft_ord[:,5]

    phenotype_ft_vec = np.zeros((num_subjects,4))
    phenotype_ft_vec_eye = np.zeros((num_subjects, 2))
    phenotype_ft_vec_hand = np.zeros((num_subjects, 3))
        
    for i in range(num_subjects):
        phenotype_ft_vec[i, int(gender[i])] = 1
        phenotype_ft_vec[i, -2] = age[i]
        phenotype_ft_vec[i, -1] = fiq[i]
        phenotype_ft_vec_eye[i, int(eye[i])] = 1
        phenotype_ft_vec_hand[i, int(hand[i])] = 1
    
    if model == 'MIDA':
        phenotype_ft_vec = np.concatenate([phenotype_ft_vec, phenotype_ft_vec_hand], axis=1)
    else:
        phenotype_ft_vec = np.concatenate([phenotype_ft_vec, phenotype_ft_vec_hand, phenotype_ft_vec_eye], axis=1)

    return phenotype_ft_vec

def get_subject_score(subject_IDs, scores, phenotype_file_path):
    """
    Get phenotype values for a list of subjects
    Args: 
        subject_IDs : list of subject IDs
        scores: list of phenotypic information to be used
        phenotype_file_path: path of the csv file including phenotypic information
    Return:
        scores_dict: phenotype values for a list of subjects
    """
    scores_dict = {}
    with open(phenotype_file_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_IDs:
                if scores == 'HANDEDNESS_CATEGORY':
                    if (row[scores].strip() == '-9999') or (row[scores].strip() ==''):
                        scores_dict[row['SUB_ID']] = 'R'
                    elif row[scores] == 'Mixed':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    elif row[scores] == 'L->R':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    else:
                        scores_dict[row['SUB_ID']] = row[scores]
                elif (scores == 'FIQ' or scores == 'PIQ' or scores == 'VIQ'):
                    if (row[scores].strip() == '-9999') or (row[scores].strip() == ''):
                        scores_dict[row['SUB_ID']] = 100
                    else:
                        scores_dict[row['SUB_ID']] = float(row[scores])

                else:
                    scores_dict[row['SUB_ID']] = row[scores]

    return scores_dict

# Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, subject_IDs, phenotype_file_path):
    """
    Args:
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_IDs : list of subject IDs
        phenotype_file_path: path of the csv file including phenotypic information
    Return:
        pheno_ft_cat : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_IDs)
    pheno_ft_cat = pd.DataFrame()
    global_phenos = []

    for i, l in enumerate(scores):
        phenos = []
        label_dict = get_subject_score(subject_IDs, l, phenotype_file_path)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                phenos.append(float(label_dict[subject_IDs[k]]))
        else:
            for k in range(num_nodes):
                phenos.append(label_dict[subject_IDs[k]])
        global_phenos.append(phenos)
    
    for i, l in enumerate(scores):
        pheno_ft_cat.insert(i, l, global_phenos[i], True)

    return pheno_ft_cat

# calculate number of domains
def get_domain_nums(pheno_ft_cat):
    """
    Args:
        pheno_ft_cat : adjacency matrix of the population graph (num_subjects x num_subjects)
    Return:
        num_domains : number of domains
    """
    num_domains = len(pheno_ft_cat['SITE_ID'].unique())
    return num_domains