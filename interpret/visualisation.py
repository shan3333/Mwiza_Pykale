"""
Top ranked feature visualisation.
Reference: https://github.com/kundaMwiza/fMRI-site-adaptation/blob/master/Biomarker_study/Consistency%20study.ipynb

"""
import numpy as np
import pandas as pd
from nilearn import plotting
from matplotlib import pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
np.random.seed(12)

class Visualisation:
    """
    Visualisation for ranking features

    Args:
        atlas_coords: atlas coordinations
        rank_df: top k ranked feature dataframe
        pos_rank_df: top k ranked positive feature dataframe
        neg_rank_df: top k ranked negative feature dataframe
        num: number of features
        abs_rank: true for ranking absolute values of features, 
                  false for positive and negative values of features
                  (default: true)
    """
    def __init__(
        self,
        atlas_coords,
        rank_df=None,
        pos_rank_df=None,
        neg_rank_df=None,
        num=10,
        abs_rank=True,
    ):
        """Init Visualisation"""
        super(Visualisation, self).__init__()
        self.atlas_coords = atlas_coords

        # visualise features of absolute values
        if abs_rank:
            self.num=num
            self.rank_df = rank_df
            self.visual_abs()
        # visualise features of positive and negative values
        else:
            self.num=num/2
            self.pos_rank_df = pos_rank_df
            self.neg_rank_df = neg_rank_df
            self.visual_pos_neg()

    def visual_abs(self):
        # top specfic-number ranked features
        for_plot_df = self.rank_df[['ROI 1 index', 'ROI 2 index', 'average weight', 'ROI 1 HO name', 'ROI 2 HO name']].head(self.num)

        inds_1 = list(for_plot_df['ROI 1 index'])
        inds_2 = list(for_plot_df['ROI 2 index'])
        inds = list(zip(inds_1, inds_2))
        top_weights = list(for_plot_df['average weight'])
        ind_to_name = dict()

        # top specific-number ranked features HO atlas labels
        for i, index in enumerate(inds):
            ind_to_name[index[0]] = for_plot_df.iloc[i, 3]
            ind_to_name[index[1]] = for_plot_df.iloc[i, 4]
            
        indices = sorted(list(set(inds_1 + inds_2)))
        index_map = dict()
        inverse_map = dict()

        for i, val in enumerate(indices):
            index_map[val] = i
            inverse_map[i] = val

        new_size = len(index_map)
        adj_mat = np.zeros(shape=(new_size, new_size))

        for i in range(10):
            adj_mat[index_map[inds_1[i]], index_map[inds_2[i]]] = top_weights[i]
            adj_mat[index_map[inds_2[i]], index_map[inds_1[i]]] = top_weights[i]

        new_atlas_cords = self.atlas_coords[indices]

        # display correlation matrix
        display_cor = plotting.plot_matrix(adj_mat ,colorbar=True)

        # edge weights visualisation
        display_edge = plotting.plot_connectome(adj_mat, new_atlas_cords, node_size=20, annotate=True)

        labels = []
        for j in range(new_size):
            labels.append(ind_to_name[inverse_map[j]])
            
        for i in range(len(new_atlas_cords)):
            display_edge.add_markers(marker_coords=[new_atlas_cords[i]], marker_color=list(np.random.rand(3,)), marker_size=30,
                                label=labels[i])

        plt.legend(bbox_to_anchor=(1.1, 1.3))

        plt.savefig('biomarkers.png', dpi=300, bbox_extra_artists=(plt.legend(bbox_to_anchor=(1.1, 1.3)),), bbox_inches='tight')
    
    def visual_pos_neg(self):
        for_plot_df_pos = self.pos_rank_df[['ROI 1 index', 'ROI 2 index', 'average weight', 'ROI 1 HO name', 'ROI 2 HO name']].head(self.num)
        for_plot_df_neg = self.neg_rank_df[['ROI 1 index', 'ROI 2 index', 'average weight', 'ROI 1 HO name', 'ROI 2 HO name']].head(self.num)

        inds_1 = list(for_plot_df_pos['ROI 1 index'])
        inds_2 = list(for_plot_df_pos['ROI 2 index'])
        inds_3 = list(for_plot_df_neg['ROI 1 index'])
        inds_4 = list(for_plot_df_neg['ROI 2 index'])
        inds_pos = list(zip(inds_1, inds_2))
        inds_neg = list(zip(inds_3, inds_4))
        ind_to_name = dict()

        top_weights_pos = list(for_plot_df_pos['average weight'])
        top_weights_neg = list(for_plot_df_neg['average weight'])

        # top specific-number positive features HO atlas labels
        for i, index in enumerate(inds_pos):
            ind_to_name[index[0]] = for_plot_df_pos.iloc[i, 3]
            ind_to_name[index[1]] = for_plot_df_pos.iloc[i, 4]

        # top specific-number negative features HO atlas labels
        for i, index in enumerate(inds_neg):
            ind_to_name[index[0]] = for_plot_df_neg.iloc[i, 3]
            ind_to_name[index[1]] = for_plot_df_neg.iloc[i, 4]

        indices = sorted(list(set(inds_1 + inds_2 + inds_3 + inds_4)))
        index_map = dict()
        inverse_map = dict()

        for i, val in enumerate(indices):
            index_map[val] = i
            inverse_map[i] = val
            
        new_size = len(index_map)
        adj_mat = np.zeros(shape=(new_size, new_size))

        for i in range(5):
            adj_mat[index_map[inds_1[i]], index_map[inds_2[i]]] = top_weights_pos[i]
            adj_mat[index_map[inds_2[i]], index_map[inds_1[i]]] = top_weights_pos[i]
            adj_mat[index_map[inds_3[i]], index_map[inds_4[i]]] = top_weights_neg[i]
            adj_mat[index_map[inds_4[i]], index_map[inds_3[i]]] = top_weights_neg[i]

        new_atlas_cords = self.atlas_coords[indices]

        labels = []

        for j in range(new_size):
            labels.append(ind_to_name[inverse_map[j]])

        # display correlation matrix
        display_cor = plotting.plot_matrix(adj_mat ,colorbar=True)

        # edge weights visualisation
        display_edge = plotting.plot_connectome(adj_mat, new_atlas_cords, node_size=20, annotate=True)

        labels = []

        for j in range(new_size):
            labels.append(ind_to_name[inverse_map[j]])
            
        for i in range(len(new_atlas_cords)):
            display_edge.add_markers(marker_coords=[new_atlas_cords[i]], marker_color=list(np.random.rand(3,)), marker_size=30,
                                label=labels[i])

        plt.legend(bbox_to_anchor=(1.1, 1.3))

        plt.savefig('biomarkers.png', dpi=300, bbox_extra_artists=(plt.legend(bbox_to_anchor=(1.1, 1.3)),), bbox_inches='tight')