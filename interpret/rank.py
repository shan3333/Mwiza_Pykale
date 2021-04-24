"""
Feature ranking for Visualisation.
Reference: https://github.com/kundaMwiza/fMRI-site-adaptation/blob/master/Biomarker_study/Consistency%20study.ipynb

"""
import pickle
import pandas as pd
import numpy as np
import re
import warnings

class Rank:
    """
    Rank features in individual folders.

    Args:
        top_k: number of features
        weights_pkl: weights generated from the specific model
        roi_labels: roi labels in the specific atlas
        abs_rank: true for ranking absolute values of features, 
                  false for positive and negative values of features
                  (default: true)
        generated_csv: true for generating csv files of the ranked features
                      (default: false)
    """
    def __init__(
        self,
        top_k=50,
        weights_pkl='LR_tangent_weights.pkl',
        roi_labels='CC200_ROI_labels.csv',
        abs_rank=True,
        generated_csv=False
    ):
        """Init Rank"""
        super(Rank, self).__init__()
        self.top_k = top_k
        self.abs_rank = abs_rank
        self.generated_csv = generated_csv

        # load weights
        with open(weights_pkl, "rb") as f:
            weights = pickle.load(f)
            self.weights = [weight*(-1) for weight in weights]

        # load ROI labels 
        self.atlas_df = pd.read_csv(roi_labels)

        # load ROI coordinates 
        temp = self.atlas_df.iloc[:, 2].map(lambda x: x.strip(')').strip('(').split(';')).tolist()
        temp = [[float(j) for j in i] for i in temp]
        self.atlas_coords = np.asarray(temp)

        ranked_feats = self.rank_features(weights)
        # rank features by absolute value
        if abs_rank:

            ranked_feats = [list(t) for t in zip(*ranked_feats)]
            feat_no = ranked_feats[0]
            ranks = ranked_feats[1]

            rank_df = pd.DataFrame(ranks, columns = ['no occurrence in top '+str(top_k), 'average weight'])

            rank_df.insert(0, 'feat_no', feat_no)
            rank_df['abs_avg_weights'] = np.absolute(rank_df['average weight'])
            rank_df = rank_df.sort_values(by=['no occurrence in top '+str(top_k), 'abs_avg_weights'], 
                                        ascending=[False, False]).reset_index().drop(['index', 'abs_avg_weights'], axis=1)

            # get feature ROI labels
            rank_df = self.get_identification(rank_df, self.atlas_df)

            # generate to csv file
            if generated_csv:
                rank_df.to_csv('top_50_features.csv')
                rank_df.head(50)

        # rank features by absolute value independently for positive and negative weights
        else:
            # ranked_feats returns two dataframes, one for the positve weight ranks
            # and another for the negative weight ranks

            pos_ranked_feats = ranked_feats[0]
            neg_ranked_feats = ranked_feats[1]
            pos_ranked_feats = [list(t) for t in zip(*pos_ranked_feats)]
            neg_ranked_feats = [list(t) for t in zip(*neg_ranked_feats)]

            pos_feat_no = pos_ranked_feats[0]
            neg_feat_no = neg_ranked_feats[0]
            pos_ranks = pos_ranked_feats[1]
            neg_ranks = neg_ranked_feats[1]

            pos_rank_df = pd.DataFrame(pos_ranks, columns = ['no occurrence in top '+str(top_k), 'average weight'])
            neg_rank_df = pd.DataFrame(neg_ranks, columns = ['no occurrence in top '+str(top_k), 'average weight'])

            pos_rank_df.insert(0, 'feat_no', pos_feat_no)
            neg_rank_df.insert(0, 'feat_no', neg_feat_no)

            pos_rank_df.loc[pos_rank_df['average weight'] <=0, 'average weight'] = 0
            neg_rank_df.loc[neg_rank_df['average weight'] >=0, 'average weight'] = 0

            pos_rank_df['abs_avg_weights'] = np.absolute(pos_rank_df['average weight'])
            neg_rank_df['abs_avg_weights'] = np.absolute(neg_rank_df['average weight'])


            pos_rank_df = pos_rank_df.sort_values(by=['no occurrence in top '+str(top_k), 'abs_avg_weights'], 
                                                ascending=[False, False]).reset_index().drop(['index', 'abs_avg_weights'], axis=1)
            neg_rank_df = neg_rank_df.sort_values(by=['no occurrence in top '+str(top_k), 'abs_avg_weights'],
                                                ascending=[False, False]).reset_index().drop(['index', 'abs_avg_weights'], axis=1)

            # get feature ROI labels for both positive and negative weight rankings
            pos_rank_df = self.get_identification(pos_rank_df)
            neg_rank_df = self.get_identification(neg_rank_df)

            # generate to csv file
            if generated_csv:
                # top 50 positive features
                pos_rank_df.to_csv('top_50_ASD_fts.csv')
                pos_rank_df.head(50)

                # top 50 negative features
                neg_rank_df.to_csv('top_50_control_fts.csv')
                neg_rank_df.head(50)

    # get atlas coordinations
    def get_atlas_coords(self):
        return self.atlas_coords

    # for a given fold (weight vector) count whether 
    # a given feature appears in the top k weights
    def get_ranks(self, weight, rank_dict):
        for ind, value in enumerate(np.absolute(weight).argsort(axis=None)):
            if value not in rank_dict:
                rank_dict[value] = [ind]
            else:
                rank_dict[value].append(ind)     

        return rank_dict

    # sort all features rank over the 10 folds 
    #(by number of occurrence)
    def sort_ranks(self, fold_weights, ranks):
        no_feats = len(ranks)
        ranks_2 = dict()
        avg_weights = np.asarray([j.flatten() for j in fold_weights]).mean(axis=0)
        for i in ranks:
            ranks_2[i] = [1 if k in list(range(no_feats-self.top_k, no_feats)) else 0 for k in ranks[i]]
            ranks_2[i] = [sum(ranks_2[i]), avg_weights[i]]
        sorted_ranks = sorted(ranks_2.items(), key=lambda x: x[1][0], reverse=True)
        return sorted_ranks

    def rank_features(self, fold_weights):
        ranks = dict()
        ranks_pos = dict()
        ranks_neg = dict()
                
        for weight in self.weights:
            weight = weight.flatten()
            if self.abs_rank != True:
                weight_pos = weight.copy()
                weight_neg = weight.copy()
                weight_pos[weight_pos <=0] = 0
                weight_neg[weight_neg >=0] = 0
                ranks_pos = self.get_ranks(weight_pos, ranks_pos)
                ranks_neg = self.get_ranks(weight_neg, ranks_neg)
            else:
                ranks = self.get_ranks(weight, ranks)

        if self.abs_rank != True:
            pos_ranks = self.sort_ranks(fold_weights, ranks_pos)
            neg_ranks = self.sort_ranks(fold_weights, ranks_neg)
            return (pos_ranks, neg_ranks)
        else:
            abs_ranks = self.sort_ranks(fold_weights, ranks)
            return abs_ranks

        
    # identify each feature by pair of ROIs number, HO atlas names
    # and centre of mass (COM)
    def get_identification(self, df, size_conn_mat=200):
        new = np.zeros((200,200))
        inds = np.triu_indices_from(new, 1)
        
        tup_index1 = []
        tup_index2 = []
        tup_COM1 = []
        tup_COM2 = []
        tup_name1 = []
        tup_name2 = []
        pattern1 = re.compile('\["[\(\)a-zA-Z\s;]+": [\d\.]*\]')
        pattern2 = re.compile('[\d:\.\[\]]+')
        
        for i in range(df.shape[0]):
            feat_no = df.iloc[i, 0]
            tup_index1.append(inds[0][feat_no])
            tup_index2.append(inds[1][feat_no])
            ind1 = inds[0][feat_no]
            ind2 = inds[1][feat_no]
            tup_COM1.append(self.atlas_df.iloc[ind1, 2])
            tup_COM2.append(self.atlas_df.iloc[ind2, 2])
            tup_name1.append(pattern2.sub('', pattern1.match(self.atlas_df.iloc[ind1, 7]).group(0)))
            tup_name2.append(pattern2.sub('', pattern1.match(self.atlas_df.iloc[ind2, 7]).group(0)))

        df['ROI 1 index'] = tup_index1
        df['ROI 2 index'] = tup_index2
        df['ROI 1 COM'] = tup_COM1
        df['ROI 2 COM'] = tup_COM2
        df['ROI 1 HO name'] = tup_name1
        df['ROI 2 HO name'] = tup_name2
        
        return df
