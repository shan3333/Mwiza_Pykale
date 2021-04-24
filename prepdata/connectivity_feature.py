"""
Preprocessing of ABIDE datasets: 
- functions for extracting connectivity features: correlation, TPE (tangent pearson), TE (tangent), ensemble (correlation + TPE + TE)
Reference:
- https://github.com/kundaMwiza/fMRI-site-adaptation/blob/97b0d26a4bf199efe82c2e1d52fcb93986eb10da/imports/preprocess_data.py
"""

import os
import numpy as np
from nilearn import connectome
import scipy.io as sio

def get_timeseries(subject_IDs, atlas_name, data_folder, silence=False):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []
    for i in range(len(subject_IDs)):
        subject_folder = os.path.join(data_folder, subject_IDs[i])
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        fl = os.path.join(subject_folder, ro_file[0])
        if silence != True:
            print("Reading timeseries file %s" %fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries

def subject_connectivity(timeseries, subjects_IDs, atlas_name, kind, data_folder, iter_no='', seed=1234, validation_ext='10CV', n_subjects='', save=True):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subjects_IDs     : subject IDs
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        data_folder    : specify path to save the matrix if different from subject folder
        iter_no      : tangent connectivity iteration number for cross validation evaluation
        save         : save the connectivity matrix to a file
    returns:
        connectivity : connectivity matrix (regions x regions)
    """
        
    if kind in ['TPE', 'TE', 'correlation']:
        if kind not in ['TPE', 'TE']:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform(timeseries)
        else:
            if kind == 'TPE':
                conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                conn_mat = conn_measure.fit_transform(timeseries)
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(conn_mat)
                connectivity = connectivity_fit.transform(conn_mat)
            else:
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(timeseries)
                connectivity = connectivity_fit.transform(timeseries)
            
    if save:
        if kind not in  ['TPE', 'TE']:
            for i, subj_id in enumerate(subjects):
                subject_file = os.path.join(data_folder, subj_id,
                                            subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
                sio.savemat(subject_file, {'connectivity': connectivity[i]})
            return connectivity
        else:
            for i, subj_id in enumerate(subjects):
                subject_file = os.path.join(data_folder, subj_id,
                                subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(iter_no) + '_' + str(seed) + '_' + validation_ext + str(n_subjects) + '.mat')
                sio.savemat(subject_file, {'connectivity': connectivity[i]})  
            return connectivity_fit


# Load precomputed fMRI connectivity networks
def get_networks(subject_IDs, kind, data_folder, iter_no='', seed=1234, validation_ext='10CV', n_subjects='', atlas_name="aal", variable='connectivity'):
    """
        subject_IDs : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks
    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_IDs:
        if len(kind.split()) == 2:
            kind = '_'.join(kind.split())
        if kind not in ['TPE', 'TE']:
            fl = os.path.join(data_folder, subject,
                            subject + "_" + atlas_name + "_" + kind.replace(' ', '_') +".mat")
        else:
            fl = os.path.join(data_folder, subject,
                    subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + '_' + str(iter_no) + '_' + str(seed) + '_' + validation_ext + str(n_subjects)+ ".mat")
            
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)
    
    if kind in ['TE', 'TPE']:
        norm_networks = [mat for mat in all_networks]
    else:
        norm_networks = [np.arctanh(mat) for mat in all_networks]

    idx = np.triu_indices_from(all_networks[0], 1)
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix