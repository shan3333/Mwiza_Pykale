from embed import ADIBE_embed
from evaluate import KHSIC_eval
import numpy as np
import pandas as pd

# load data
subject_IDs = []
num_classes = 0
labels = dict()
data_folder = ''
seed = 0

# prepdata
num_domains = 0
model = ''
algorithm = ''
phenotype_ft = []
phenotype_raw = []
connectivity = ''
atlas = ''
phenotypes = True
is_KHSIC = True
filename = ''

# embed
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

