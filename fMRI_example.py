from embed import ADIBE_embed
from evaluate import KHSIC_eval
import numpy as np
import pandas as pd
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. '
                                                 'MIDA is used to minimize the distribution mismatch between ABIDE sites')
    parser.add_argument('--atlas', default='cc200',
                        help='Atlas for network construction (node definition) options: ho, cc200, cc400, default: cc200.')
    parser.add_argument('--model', default='MIDA', type=str, help='Options: MIDA, raw. default: MIDA.')
    parser.add_argument('--algorithm', default='Ridge', type=str, help='Options: Ridge, LR (Logistic regression),'
                                                                       ' SVM (Support vector machine). default: Ridge.')
    parser.add_argument('--phenotypes', default=True, type=str2bool, help='Add phenotype features. default: True.')
    parser.add_argument('--KHSIC', default=True, type=str2bool,
                        help='Compute kernel statistical test of independence between features'
                             ' and site, default True.')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    parser.add_argument('--connectivity', default='TPE', type=str, help='Type of connectivity used for network '
                                                                        'construction. options: correlation, TE(tangent embedding), TPE(tangent pearson embedding),'
                                                                        'default: TPE.')
    parser.add_argument('--leave_one_out', default=False, type=str2bool,
                        help='leave one site out CV instead of 10CV. default: False.')
    parser.add_argument('--filename', default='tangent', type=str, help='filename for output file. default: tangent.')
    parser.add_argument('--ensemble', default=False, type=str2bool,
                        help='Leave one site out, use ensemble MIDA/raw. default: False')

    args = parser.parse_args()
    print('Arguments: \n', args)

    # load data
    subject_IDs = [] # =======================
    num_classes = 2
    labels = dict() # =======================
    data_folder = ''  # =======================
    seed = args.seed                    # seed for random initialisation

    # prepdata
    num_domains = 0  # =======================
    model = args.model                  # MIDA, SMIDA or raw
    algorithm = args.algorithm
    phenotype_ft = []  # =======================
    phenotype_raw = []  # =======================
    connectivity = args.connectivity    # Type of connectivity used for network construction
    atlas = args.atlas                  # Atlas for network construction
    phenotypes = args.phenotypes        # Add phenotype features
    is_KHSIC = args.KHSIC               # Compute kernel statistical test of independence between features and site (boolean)
    filename = args.filename            # Results output file
    leave_one_out = args.leave_one_out
    ensemble = args.ensemble

    # embed
    if leave_one_out == True:
        if ensemble == True:
            train = ADIBE_embed.TrainABIDE(subject_IDs, num_classes, labels, data_folder, num_domains, model, algorithm)
            results_acc, results_auc, features, domain_ft = train.leave_one_site_out_ensemble(phenotype_ft,
                                                                                              phenotype_raw,
                                                                                              seed, atlas, phenotypes)
        else:
            train = ADIBE_embed.TrainABIDE(subject_IDs, num_classes, labels, data_folder, num_domains, model, algorithm)
            results_acc, results_auc, features, domain_ft = train.leave_one_site_out(phenotype_ft, phenotype_raw,
                                                                                     seed, atlas, phenotypes)
    else:
        train = ADIBE_embed.TrainABIDE(subject_IDs, num_classes, labels, data_folder, num_domains, model, algorithm)
        results_acc, results_auc, features, domain_ft = train.train_10CV(phenotype_ft, phenotype_raw, seed,
                                                                         connectivity, atlas, phenotypes)

    # evaluation
    avg_acc = np.array(results_acc).mean()
    std_acc = np.array(results_acc).std()
    avg_auc = np.array(results_auc).mean()
    std_auc = np.array(results_auc).std()
    print("accuracy average", avg_acc)
    print("standard deviation accuracy", std_acc)
    print("auc average", avg_auc)
    print("standard deviation auc", std_auc)

    if leave_one_out == False:
        if is_KHSIC == True and model == 'MIDA':
            KHSIC_model = KHSIC_eval.KHSIC(features, domain_ft)
            test_stat, threshold, pval = KHSIC_model.hsic_gam(alph=0.05)
            pval = 1 - pval
            print('KHSIC sample value: %.2f' % test_stat, 'Threshold: %.2f' % threshold, 'p value: %.10f' % pval)

    all_results = pd.DataFrame()
    all_results['ACC'] = results_acc
    all_results['AUC'] = results_auc
    all_results.to_csv(filename + '.csv')

    # interpret


if __name__ == '__main__':
    main()
