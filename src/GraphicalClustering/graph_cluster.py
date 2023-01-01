import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.clique import find_cliques
from collections import Counter
from numpy.core.defchararray import find
from collections import OrderedDict
from collections import namedtuple

warnings.filterwarnings('ignore')
root_dir = os.path.abspath(os.path.dirname('__file__'))
os.chdir(root_dir)

class GraphicalClustering:

    def __init__(self, input_text, min_cluster_members):
        self.input_file = input_text
        self.min_cluster_members = min_cluster_members
        self.input_file = self.input_file.astype(object).where(pd.notnull(self.input_file), '')
        self.cluster_id = []
        self.cluster_label = []
        self.cluster_threshold = []
        self.cluster_data = pd.DataFrame()

    @staticmethod
    def get_processed_input(input_file):
        input_file_df = pd.DataFrame(input_file.values, columns=['ActualText'])
        # Converting data to lower case
        input_file_df['ProcessedText'] = input_file_df['ActualText'].astype(str).str.lower()
        # Replace _ with ''
        input_file_df['ProcessedText'] = (input_file_df['ProcessedText']).str.replace('_', '')
        # Removal special character expect alpha and numeric
        input_file_df['ProcessedText'] = (input_file_df['ProcessedText']).str.replace('[^A-Za-z0-9 ]+', ' ')
        input_file_df['ProcessedText'] = (input_file_df['ProcessedText']).str.replace(r'\b\w{1,2}\b', '')
        # Remove numeric except alpha- numeric data
        input_file_df['ProcessedText'] = (input_file_df['ProcessedText']).str.replace(r'\b[0-9]+\b', '')
        input_file_df['ProcessedText'] = input_file_df['ProcessedText'].str.strip()
        input_file_df['Row_Number'] = range(0, len(input_file_df.index))
        return input_file_df

    def preprocessing(self, input_file_):
        input_file = self.get_processed_input(input_file_)
        input_file.sort_values("ProcessedText", inplace=True)
        empty_df = input_file.loc[input_file["ProcessedText"] == ""]
        empty_df['FinalRowIndex'] = ""
        empty_df['Cluster_ID'] = 0
        empty_df['Cluster_Label'] = ''
        if len(empty_df) == 0:
            empty_df['threshold'] = None
        else:
            empty_df['threshold'] = 0
        nonempty_df = input_file.loc[input_file["ProcessedText"] != ""]
        nonempty_df['RowIndex'] = range(0, len(nonempty_df.index))
        nonempty_df['FinalRowIndex'] = range(0, len(nonempty_df.index))
        nonempty_df.index = range(0, len(nonempty_df.index))
        nonempty_df['split_tokens_original'] = nonempty_df['ProcessedText'].apply(str.split)
        nonempty_df['split_tokens'] = nonempty_df['split_tokens_original'].apply(set)
        nonempty_df['split_tokens'] = nonempty_df['split_tokens'].apply(sorted)
        nonempty_df['split_tokens'] = nonempty_df['split_tokens'].apply(set)
        nonempty_df['split_tokens_new'] = nonempty_df['split_tokens'].apply(lambda x: ' '.join(map(str, x)))
        return empty_df, nonempty_df

    def grouping_clustering(self, nonempty_df):
        group_df = nonempty_df.groupby(['split_tokens_new']).size().reset_index(name='counts')
        group_df = group_df.loc[group_df['counts'] >= self.min_cluster_members]
        group_df.sort_values("counts", ascending=False, inplace=True)
        group_df['Cluster_ID'] = range(1, len(group_df) + 1)
        group_df['threshold'] = 1
        group_df_col = ['split_tokens_new', 'Cluster_ID', 'threshold']
        rev_nonempty_df = nonempty_df.merge(group_df[group_df_col], on='split_tokens_new', how='left')
        # Cluster Data
        cluster_set = rev_nonempty_df.loc[rev_nonempty_df.Cluster_ID.notnull()]
        cluster_set = cluster_set[['FinalRowIndex', 'Cluster_ID', 'threshold']]
        # Non Cluster Data
        non_cluster_set = rev_nonempty_df.loc[~rev_nonempty_df.Cluster_ID.notnull()]
        cluster_id = len(group_df)
        return cluster_set, non_cluster_set, cluster_id

    @staticmethod
    def compute_issue_adjacency_matrix(desc_input, threshold_limit):
        # Compute dice similarity for each pair of cleaned description
        issue_adjacency_matrix = list(desc_input['split_tokens']. \
                                      apply(lambda y: list(map(lambda x: \
                                                                   (2 * len(set(y).intersection(set(x)))) \
                                                                   / (len(set(y)) + len(set(x))), \
                                                               desc_input['split_tokens']))))
        # Store the issue Adjacency Matrix as numpy array for better use
        data = np.asarray(issue_adjacency_matrix)
        # filter the pair which are having >=70% text similarity
        dice_df = (data >= threshold_limit)
        return dice_df

    @staticmethod
    def get_maximum_list(row, col, max_no):
        # Getting the cliques or clusters having the maximum count of elements
        tuple_list = list(zip(row, col))
        graph = nx.Graph()
        graph.add_edges_from(tuple_list)
        lst = list(find_cliques(graph))
        clq_list = max(lst, key=lambda i: len(i))
        if len(clq_list) < max_no:
            clq_list = []
        return clq_list

    def clustering(self, dice_df, nonempty_df, threshold, cluster_id):
        cluster_df = pd.DataFrame(columns=["RowIndex", "Cluster_ID", "threshold"])
        row, col = np.where(dice_df == True)
        row = list(row)
        col = list(col)
        # CLUSTER ID GENERATION
        while True:
            max_list = self.get_maximum_list(row, col, self.min_cluster_members)
            max_len = len(max_list)
            if max_len != 0:
                cluster_id += 1
                temp_cluster = pd.DataFrame(
                    data={"RowIndex": nonempty_df.loc[nonempty_df['RowIndex'].isin(max_list), 'FinalRowIndex'],
                          "Cluster_ID": [cluster_id] * max_len, "threshold": [threshold] * max_len})
                cluster_df = cluster_df.append(temp_cluster)
                col = list(pd.Series(col)[list(pd.Series(row)[~pd.Series(row).isin(max_list)].index)])
                row = list(pd.Series(row)[~pd.Series(row).isin(max_list)])
                row = list(pd.Series(row)[list(pd.Series(col)[~pd.Series(col).isin(max_list)].index)])
                col = list(pd.Series(col)[~pd.Series(col).isin(max_list)])
                if (len(row) == 0) | (len(col) == 0):
                    break
            else:
                break
        cluster_df.RowIndex = cluster_df.RowIndex.map(int)
        return cluster_df

    def graphical_clustering(self, cluster_set, non_cluster_set, threshold, cluster_id):
        non_cluster_set["RowIndex"] = range(0, len(non_cluster_set.index))
        dice_df = self.compute_issue_adjacency_matrix(non_cluster_set, threshold)
        cluster_df_set = self.clustering(dice_df, non_cluster_set, threshold, cluster_id)
        cluster_df_set.columns = ['FinalRowIndex', 'Cluster_ID', 'threshold']
        cluster_set = cluster_set.append(cluster_df_set)
        return cluster_set, cluster_df_set

    @staticmethod
    def get_cluster_model(nonempty_df_set, cluster_df_set):
        temp_cls_data = nonempty_df_set.loc[nonempty_df_set.FinalRowIndex.isin(cluster_df_set.FinalRowIndex), :]
        temp_cls_data["RowIndex"] = range(0, len(temp_cls_data.index))
        temp_cls_data.drop(['Cluster_ID'], axis=1, inplace=True)
        temp_cls_data = temp_cls_data.merge(cluster_df_set[['FinalRowIndex', 'Cluster_ID']], on='FinalRowIndex',
                                            how='left')
        # Get Cluster Definition non_cluster_df_set
        clustered_group = temp_cls_data.groupby(["Cluster_ID"])["RowIndex"].aggregate(lambda x: set(x)).reset_index()
        clustered_group["TicketsCount"] = clustered_group["RowIndex"].apply(len)
        clustered_group = clustered_group.sort_values('TicketsCount', ascending=False)
        return temp_cls_data, clustered_group

    @staticmethod
    def calculate_dice_coefficient(a, *args):
        # Calculating dice coefficient
        c = set(args)
        value = (2 * len(a.intersection(c))) / (len(a) + len(c))
        return value

    @staticmethod
    def find_subset(set1, *args):
        set2 = set(args)
        return set1.issubset(set2)

    def get_cluster_prediction(self, temp_cls_data, clustered_group, non_cluster_set, cluster_set, threshold):
        for i in non_cluster_set.index:
            temp_array = np.array(
                temp_cls_data['split_tokens'].apply(self.calculate_dice_coefficient,
                                                    args=non_cluster_set["split_tokens"][i]))
            temp_array = temp_array >= threshold
            indexes = np.array(range(len(temp_array)))
            possible_clusters = list(np.array(clustered_group['Cluster_ID'])[
                                         clustered_group["RowIndex"].apply(self.find_subset,
                                                                           args=set((indexes[temp_array])))])
            if len(possible_clusters) >= 1:
                possible_clusters = [possible_clusters[0]]
                non_cluster_set.loc[i, 'Cluster_ID'] = possible_clusters
                non_cluster_set.loc[i, 'threshold'] = threshold
        predict_set = non_cluster_set.loc[non_cluster_set.Cluster_ID.notnull(), :]
        predict_set = predict_set[['FinalRowIndex', 'Cluster_ID', 'threshold']]
        cluster_set = cluster_set.append(predict_set)
        return non_cluster_set, cluster_set

    def iterative_clustering(self, non_cluster_df_set, cluster_set, non_predict_set, threshold):
        uc_df_set_bkp = non_cluster_df_set
        for i in non_cluster_df_set.index:
            if sum(cluster_set.FinalRowIndex.isin([non_cluster_df_set['FinalRowIndex'][i]])) < 1:
                req_col = ['FinalRowIndex', 'split_tokens']
                uc_df_set_bkp = uc_df_set_bkp.loc[
                    ~uc_df_set_bkp.FinalRowIndex.isin([non_cluster_df_set['FinalRowIndex'][i]]), req_col]
                temp_non_predict_set = non_predict_set[req_col]
                temp_non_predict_set = temp_non_predict_set.append(uc_df_set_bkp)
                temp_non_predict_set['dice_metric'] = temp_non_predict_set['split_tokens'].apply(
                    self.calculate_dice_coefficient, args=non_cluster_df_set["split_tokens"][i])
                itr_cluster_set = temp_non_predict_set.loc[temp_non_predict_set.dice_metric >= threshold, :]
                max_len = len(itr_cluster_set)
                if max_len >= self.min_cluster_members - 1:
                    itr_cluster_id = cluster_set.Cluster_ID.max() + 1
                    temp_cluster = pd.DataFrame(data={"FinalRowIndex": itr_cluster_set.FinalRowIndex.to_list(),
                                                      "Cluster_ID": [itr_cluster_id] * max_len,
                                                      "threshold": [threshold] * max_len})
                    temp_cluster.loc[len(temp_cluster.index)] = [non_cluster_df_set.FinalRowIndex[i], itr_cluster_id,
                                                                 threshold]
                    cluster_set = cluster_set.append(temp_cluster)
                    non_predict_set = non_predict_set.loc[
                                      ~non_predict_set.FinalRowIndex.isin(cluster_set.FinalRowIndex), :]
                    uc_df_set_bkp = uc_df_set_bkp.loc[~uc_df_set_bkp.FinalRowIndex.isin(cluster_set.FinalRowIndex), :]
        return cluster_set, non_predict_set, non_cluster_df_set

    @staticmethod
    def cluster_labeling(final_df):
        # Initialize the cluster label
        final_df['Cluster_Label'] = ''
        # Get unique clusters formed to generate label for the same
        unique_clusters = final_df['Cluster_ID'].unique()
        # Iterate the cluster
        for cluster_id in unique_clusters:
            if cluster_id != 0:
                # Get the text for the unique cluster
                cluster_text = final_df['ProcessedText'][final_df.Cluster_ID == cluster_id]
                # Remove duplicate words from the test
                cluster_text = cluster_text.str.split().apply(lambda x: ' '.join(OrderedDict.fromkeys(x).keys()))
                # Get unique words and its occurrence in the text
                word_count = cluster_text.str.split(expand=True).stack().value_counts()
                word_count = word_count.to_frame('word_frequency')
                word_count = word_count.reset_index()
                # Identify words which present at least in 50% of the texts
                common_words = word_count['index'][word_count.word_frequency >= len(cluster_text) * 0.5].to_list()
                final_position = []
                # Iterate the words
                for word in common_words:
                    # Identify all position's of the word
                    position_array = find(cluster_text.values.astype(str),
                                          np.full(shape=len(cluster_text), fill_value=word))
                    # Identify most common position of the word
                    final_position.append(Counter(position_array[position_array != -1]).most_common(1)[0][0])
                # Create label based on the position's identified
                label_df = pd.DataFrame(data={'Word': common_words, 'Position': final_position})
                cluster_label = label_df.sort_values(by='Position')['Word'].str.cat(sep=' ')
                # Tag the label to those cluster text
                final_df.loc[final_df.Cluster_ID == cluster_id, 'Cluster_Label'] = cluster_label
        return final_df

    def generate_graph_clusters(self, sme_batch_size):
        input_file = self.input_file
        # Preprocessing
        empty_df, nonempty_df = self.preprocessing(input_file)
        # Clustering for 100% Similarity
        cluster_set, non_cluster_set, cluster_id = self.grouping_clustering(nonempty_df)
        threshold_list = [0.9, 0.8, 0.7, 0.6, 0.5]
        if (len(non_cluster_set) <= sme_batch_size) and (len(non_cluster_set) > 0):
            for threshold in threshold_list:
                cluster_set, cluster_df_set = self.graphical_clustering(cluster_set, non_cluster_set,
                                                                        threshold, cluster_id)
                non_cluster_set = non_cluster_set.loc[~non_cluster_set.FinalRowIndex.isin(cluster_set.FinalRowIndex), :]
                if len(cluster_set) > 0:
                    cluster_id = cluster_set.Cluster_ID.max()
                if len(non_cluster_set) < self.min_cluster_members:
                    break
        elif len(non_cluster_set) > sme_batch_size:
            for threshold in threshold_list:
                backlog_count = len(non_cluster_set)
                batch_size = sme_batch_size  # Need to put this inside if condition of check whether backlog_count > 500
                # Get batch of input data based on batch size
                final_non_cluster_df = pd.DataFrame()
                while backlog_count > 0:
                    nonempty_df_set = non_cluster_set.head(batch_size)
                    cluster_set, cluster_df_set = self.graphical_clustering(cluster_set, nonempty_df_set, threshold,
                                                                            cluster_id)
                    # Get non cluster batch output
                    non_cluster_df_set = nonempty_df_set.loc[
                                         ~nonempty_df_set.FinalRowIndex.isin(cluster_df_set.FinalRowIndex), :]
                    non_cluster_set = non_cluster_set.loc[
                                      ~non_cluster_set.FinalRowIndex.isin(nonempty_df_set.FinalRowIndex), :]
                    temp_cls_data, clustered_group = self.get_cluster_model(nonempty_df_set, cluster_df_set)
                    # Get Predicted Cluster for the non cluster data
                    if len(non_cluster_set) > 0:
                        if len(clustered_group) > 0:
                            non_cluster_set, cluster_set = self.get_cluster_prediction(temp_cls_data, clustered_group,
                                                                                       non_cluster_set, cluster_set,
                                                                                       threshold)
                        # Get Non-Predicted Cluster for the non cluster data
                        non_predict_set = non_cluster_set.loc[~non_cluster_set.Cluster_ID.notnull(), :]
                        # Perform Iterative Cluster using non_cluster_df_set on non_predict_set
                        cluster_set, non_predict_set, non_cluster_df_set = self.iterative_clustering(non_cluster_df_set,
                                                                                                     cluster_set,
                                                                                                     non_predict_set,
                                                                                                     threshold)
                        non_cluster_df_set = non_cluster_df_set.loc[
                                             ~non_cluster_df_set.FinalRowIndex.isin(cluster_set.FinalRowIndex), :]
                        non_cluster_set = non_predict_set
                    final_non_cluster_df = final_non_cluster_df.append(non_cluster_df_set)
                    backlog_count = len(non_cluster_set)
                    if len(cluster_set) > 0:
                        cluster_id = cluster_set.Cluster_ID.max()
                    if backlog_count < sme_batch_size:
                        batch_size = backlog_count
                non_cluster_set = final_non_cluster_df
        # Labeling
        nonempty_df_update = nonempty_df.merge(cluster_set, on='FinalRowIndex', how='left')
        final_df = self.cluster_labeling(nonempty_df_update)
        final_df = final_df[empty_df.columns].append(empty_df)
        final_df = final_df.sort_values(by=['Row_Number'])
        self.cluster_id = final_df['Cluster_ID'].tolist()
        self.cluster_label = final_df['Cluster_Label'].tolist()
        self.cluster_threshold = final_df['threshold'].tolist()
        self.cluster_data = final_df[['ProcessedText', 'Cluster_ID', 'Cluster_Label', 'threshold']]

    def predict(self, input_file):
        cluster_data = self.cluster_data
        empty_df, nonempty_df = self.preprocessing(input_file)
        empty_df['Cluster_ID'] = 0
        # nonempty_df['S_No'] = list(range(0, len(nonempty_df)))
        cluster_data['split_tokens_original'] = cluster_data['ProcessedText'].apply(str.split)
        cluster_data['split_tokens'] = cluster_data['split_tokens_original'].apply(set)
        cluster_data['split_tokens'] = cluster_data['split_tokens'].apply(sorted)
        cluster_data['split_tokens'] = cluster_data['split_tokens_original'].apply(set)
        cluster_data_head = cluster_data.groupby('Cluster_ID').head(1)
        mod_cluster_data_head = cluster_data_head[['Cluster_ID', 'ProcessedText',
                                               "threshold", 'split_tokens']]
        cluster_data_identical = mod_cluster_data_head[(mod_cluster_data_head['threshold'] == 1)]
        predict_nonempty_df = pd.merge(nonempty_df, cluster_data_identical[['Cluster_ID', 'ProcessedText']],
                                       how='inner', on=['ProcessedText'])
        rem_nonempty_df = nonempty_df[(~nonempty_df.Row_Number.isin(predict_nonempty_df.Row_Number))]
        rem_nonempty_df['Cluster_ID'] = 0
        for threshold in [0.9, 0.8, 0.7, 0.6, 0.5]:
            if len(rem_nonempty_df) > 0:
                unique_clusters = cluster_data[(cluster_data['threshold'] == threshold)]['Cluster_ID'].unique()
                for cluster in unique_clusters:
                    if len(rem_nonempty_df) > 0:
                        tmp_cluster_data = cluster_data[(cluster_data['threshold'] == threshold) &
                                                        (cluster_data['Cluster_ID'] == cluster)]
                        temp_array = np.array([[(2 * len(x.intersection(y))) / (len(x) + len(y)) \
                                                for y in tmp_cluster_data['split_tokens']] \
                                               for x in rem_nonempty_df["split_tokens"]])
                        dice_coefficient_flag = (temp_array >= threshold).sum(axis=1) == len(tmp_cluster_data)
                        rem_nonempty_df['dice_coefficient_col'] = dice_coefficient_flag.tolist()
                        rem_nonempty_df.loc[(rem_nonempty_df['dice_coefficient_col'] == True), 'Cluster_ID'] = cluster
                        rem_nonempty_df.drop(['dice_coefficient_col'], axis=1, inplace=True)
                        predicted_data = rem_nonempty_df[(rem_nonempty_df['Cluster_ID'] != 0)]
                        rem_nonempty_df = rem_nonempty_df[(~rem_nonempty_df.Row_Number.isin(predicted_data.Row_Number))]
                        predict_nonempty_df = predict_nonempty_df.append(predicted_data)
                    else:
                        break
            else:
                break
        nonempty_df = predict_nonempty_df.append(rem_nonempty_df)
        final_df = nonempty_df.append(empty_df)
        final_df = final_df.sort_values(by=['Row_Number'])
        final_df = final_df.drop(['Cluster_Label', 'threshold'], axis=1)
        final_df = pd.merge(final_df, cluster_data_head[['Cluster_ID',
                                                    'Cluster_Label',
                                                    'threshold']],
                            how='left', on=['Cluster_ID'])
        cluster_id = list(map(int, final_df['Cluster_ID'].tolist()))
        cluster_label = final_df['Cluster_Label'].tolist()
        cluster_threshold = final_df['threshold'].tolist()
        result = namedtuple('Result', ['cluster_id',
                                       'cluster_label', 'threshold'])
        return result(cluster_id, cluster_label, cluster_threshold)
