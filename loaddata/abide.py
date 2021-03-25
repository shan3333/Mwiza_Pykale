"""
Dataset setting and data loader for ABIDE, from
http://preprocessed-connectomes-project.org/abide/
Reference: https://github.com/kundaMwiza/fMRI-site-adaptation/blob/master/fetch_data.py

"""

import logging
import os
import shutil
import csv
import numpy as np
import glob

from nilearn import datasets
from prepdata.connectivity_feature import get_timeseries, subject_connectivity, get_networks

class ABIDE:
    """
    ABIDE Dataset.
    Auto-downloads the dataset.

    Args:
        root (str): path to directory where the ABIDE folder will be created (or exists.)
        subject_IDs_root (str): root of subject_IDs.txt.
        pipeline (str): defaults to cpac
            Pipeline to preprocess ABIDE data. Available options are ccs, cpac, dparsf and niak.
        atlas (str): defaults to cc200
            Brain parcellation atlas. Options: ho, cc200 and cc400.
        download (str2bool): defaults to True
            Dowload data.
    """
    def __init__(
        self,
        root,
        subject_IDs_root,
        pipeline='cpac',
        atlas='cc200',
        download=True,
        connectivity='correlation'
    ):
        """Init ABIDE dataset."""
        super(ABIDE, self).__init__()
        self.root = root
        self.data_folder = os.path.join(root, 'ABIDE_pcp/' + pipeline + '/filt_noglobal/')
        self.pipeline = pipeline
        self.atlas = atlas
        self.subject_IDs_root = subject_IDs_root
        self.subject_IDs = self.get_ids().tolist()

        # download data
        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")
        
        # compute and save connectivity matrices
        time_series = get_timeseries(self.subject_IDs, self.atlas)
        subject_connectivity(time_series, self.subject_IDs, self.atlas, connectivity)


    def _check_exists(self):
        return os.path.exists(self.data_folder)
    
    def _check_csv_exists(self):
        return os.path.exists(os.path.join(self.root, 'ABIDE_pcp/*.csv'))
    
    # Get the list of subject IDs
    def get_ids(self, num_subjects=None):
        """
        return:
            subject_IDs    : list of all subject IDs
        """
        subject_IDs = np.genfromtxt(os.path.join(self.data_folder, 'subject_IDs.txt'), dtype=str)

        if num_subjects is not None:
            subject_IDs = subject_IDs[:num_subjects]

        return subject_IDs

    # Get class labels
    def get_labels(self, subject_IDs):
        """
        return:
            class labels: dict of labels of all subject
        """
        if self._check_csv_exists:
            scores_dict = {}
            f = glob.glob(os.path.join(self.root, 'ABIDE_pcp/*.csv'))[0]
            with open(f) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    scores_dict[row['SUB_ID']] = row['DX_GROUP']
            return scores_dict
        else:
            raise RuntimeError("CSV file not found." + " You can use download=True to re-download the dataset.")
    
    def fetch_filenames(self, subject_IDs, file_type, atlas):
        """
            subject_list : list of short subject IDs in string format
            file_type    : must be one of the available file types
            filemapping  : resulting file name format

        returns:

            filenames    : list of filetypes (same length as subject_list)
        """

        filemapping = {'func_preproc': '_func_preproc.nii.gz',
                    'rois_' + atlas: '_rois_' + atlas + '.1D'}
        # The list to be filled
        filenames = []

        # Fill list with requested file paths
        for i in range(len(subject_IDs)):
            os.chdir(self.data_folder)
            try:
                try:
                    os.chdir(self.data_folder)
                    filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
                except:
                    os.chdir(self.data_folder + '/' + subject_IDs[i])
                    filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
            except IndexError:
                filenames.append('N/A')
        return filenames

    def download(self):
        """
        Download the ABIDE data:

        """
        # fetch files
        files = ['rois_' + self.atlas]
        filemapping = {'func_preproc': 'func_preproc.nii.gz',
                       files[0]: files[0] + '.1D'}

        # check if dataset already exists
        if not os.path.exists(self.data_folder): os.makedirs(self.data_folder)
        shutil.copyfile(self.subject_IDs_root, os.path.join(self.data_folder, 'subject_IDs.txt'))

        # Download database files
        datasets.fetch_abide_pcp(data_dir=self.root, pipeline=self.pipeline,
                                band_pass_filtering=True, global_signal_regression=False, derivatives=files, quality_checked=False)

        # process and save as files
        logging.info("Processing...")

        # Create a folder for each subject
        for s, fname in zip(self.subject_IDs, self.fetch_filenames(self.subject_IDs, files[0], self.atlas)):
                subject_folder = os.path.join(self.data_folder, s)     
                if not os.path.exists(subject_folder):
                        os.mkdir(subject_folder)

                # Get the base filename for each subject
                base = fname.split(files[0])[0]

                # Move each subject file to the subject folder
                for fl in files:
                        if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
                                shutil.move(base + filemapping[fl], subject_folder)

        # Get class labels
        labels = self.get_labels(self.subject_IDs)
        # Save class labels
        with open(os.path.join(self.data_folder, 'subject_labels.txt'), "w") as f:
                print(labels, file=f)

        logging.info("[DONE]")



