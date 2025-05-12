import os
# Enforce CPU Usage
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Uncommenting enforces CPU usage  # Commenting enforces GPU usage

# Seed the Random-Number-Generator in a bid to get 'Reproducible Results'
import tensorflow as tf
from keras import backend as K
from numpy.random import seed
seed(1)
tf.compat.v1.set_random_seed(3)

import pandas as pd
import numpy as np
import math, time
import igraph as igh
import scipy.stats as stats
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.layers import Embedding, Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras import regularizers
from keras import initializers, losses, metrics, optimizers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from spektral.layers import GATConv
from tensorflow.keras.layers import Input

# Import classes from my custom package
from custom_classes.Starter_Module_01 import Starter


def sigmoid(x):
    x = np.array(x)
    y = 1/(1 + np.exp(-x))
    return y

def softmax(x):
    x = np.array(x)
    y = np.exp(x)
    z = y/y.sum()
    return z

def swish(x):
    x = np.array(x)
    y = x/(1-np.exp(-x))
    return y

def euclidean_norm(x, y):
    x = np.array(x)
    y = np.array(y)
    
    a = x - y
    b = pow(a, 2)
    c = np.sum(b)
    
    return math.sqrt(c)

def cosine_sim(x, y):
    x = np.array(x)
    y = np.array(y)
    
    a = np.sum(x * y)
    
    b = pow(x, 2)
    c = np.sum(b)
    d = math.sqrt(c)
    
    g = pow(y, 2)
    h = np.sum(g)
    i = math.sqrt(h)
    
    return a/(d * i)

def common_neigh(x, y):
    x = np.array(x)
    y = np.array(y)
    
    a = np.sum(x * y)
    
    return a
    
def args_parse_cmd():
    parser = ArgumentParser(description='START-HELP: Program for forecasting/predicting breakup or schism in social networks', epilog='END-HELP: End of assistance/help section',
                            formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('-rp', '--root_path', nargs='+', default='generic_datasets/', type=str, help='Generic root path for application/program')    
    parser.add_argument('-el', '--edge_list', nargs='+', default='Cora', type=str, help='Edge list (filename WITHOUT extension) of reference graph')  #'CiteSeer', 'Cora', 'Internet-Industry-Partnerships', 'PubMed-Diabetes', 'Terrorists-Relation', 'Zachary-Karate'
    parser.add_argument('-rm', '--run_mode', nargs='+', default='single', type=str, choices=['single', 'all'], help='Run model per specified dataset OR cumulatively for all intrinsic datasets')
    args = parser.parse_args()
    return args

def nodelist_edgelist_compute(myCls, args, edge_list):
    # Load edge list
    graph_fname = edge_list + '/' + edge_list
    df_cln2 = myCls.load_data(args.root_path, graph_fname+'.edgelist', sep='\s', header=None, index_col=None, mode='READ')
    df_cln2 = df_cln2.astype('int32')  
    temp_1 = df_cln2.values[:,:]  # edge list (NUMPY array)
    temp_2 = np.unique(temp_1)  # node list (NUMPY array)
    return temp_1, temp_2
    
def edgelist_scan(u, v, edges):
    search = str(u) + '-' + str(v)
    rtn_val = False
    if search in edges:
        rtn_val = True
    return rtn_val

def paths_compute(u, v, truth_ties, graph):
    bridge = 0
    res = False
    if u != v:
        srch_res = edgelist_scan(u, v, list(truth_ties))
        if srch_res == False:
            k_paths = graph.get_all_shortest_paths(u, v)
            #print(k_paths)  # delete
            if len(k_paths) <= bridge:
                res = [u, v, 0]
    return res

def falsehood_ties_gen(graph, nodes, edges):
    truth_ties = pd.Series(edges[:,0]).map(str) + '-' + pd.Series(edges[:,1]).map(str)
    false_ties = pd.DataFrame()
    
    for u in nodes:
        for v in nodes:
            false_row = paths_compute(u, v, truth_ties, graph)
            if false_row != False:
                false_ties = false_ties.append([false_row], ignore_index=True)
                #print(false_row)  # delete
            if false_ties.shape[0] >= truth_ties.shape[0]:
                break
        if false_ties.shape[0] >= truth_ties.shape[0]:
            break        
    return false_ties

def predHits(truth, pred1, pred2, pred3):
    hits_1 = 0
    hits_3 = 0
    pred1 = np.rint(pred1).astype(np.int32)
    pred2 = np.rint(pred2).astype(np.int32)
    pred3 = np.rint(pred3).astype(np.int32)
    
    for i in range(len(truth)):
        if truth[i] == pred1[i]:
            hits_1 = hits_1 + 1
        if (truth[i] == pred1[i]) or (truth[i] == pred2[i]) or (truth[i] == pred3[i]):
            hits_3 = hits_3 + 1
    top_1 = hits_1/len(truth)
    top_3 = hits_3/len(truth)
    
    return top_1, top_3

def evaluations(test_y, pred_y, pred_y_proba, pred_y_2, pred_y_3):
    # Evalute results via ML standards
    avg_pr = average_precision_score(test_y, pred_y_proba)
    precision = precision_score(test_y, pred_y, average='binary')
    recall = recall_score(test_y, pred_y, average='binary')
    accuracy = accuracy_score(test_y, pred_y)
    f1 = f1_score(test_y, pred_y, average='binary')
    mcc = matthews_corrcoef(test_y, pred_y)
    auc_roc = roc_auc_score(test_y, pred_y_proba)
    top_1, top_3 = predHits(test_y, pred_y, pred_y_2, pred_y_3)   
    
    print("\nLink Prediction Evaluation Report:")
    evals = {'avg_pr':round(avg_pr, 4), 'precision':round(precision, 4), 'recall':round(recall, 4), 'accuracy':round(accuracy, 4), 'f1':round(f1, 4), 'mcc':round(mcc, 4), 'auc_roc':round(auc_roc, 4), 'top_1':round(top_1, 4), 'top_3':round(top_3, 4)}
    
    return evals

def embeddings_gen(myCls, train_X, train_y, test_X, test_y, input_enc_dim, nodes, fname):
    # Hyperparameters
    repeats = 1  # 100
    n_epochs = 100  # 135    
    output_dim = 256
    input_len = 2
    
    # Implementing MODEL via Multiple Repeats OR Multiple Restarts
    for r in range(repeats):
        # Fit the Network
        start_time = time.time()  # START: Training Time Tracker    
        K.clear_session()  # Kills current TF comp-graph & creates a new one

        model = Sequential()
        model.add(Input(shape=(input_enc_dim, input_len)))
        model.add(GATConv(output_dim, attn_heads=8, concat_heads=True, activation='relu'))

        # Removed the Regression Layer and Activation ('sigmoid')
        model.add(Dense(1, kernel_initializer='glorot_uniform', 
                 kernel_regularizer=regularizers.l2(0.01), use_bias=False))  # Removed Regression Layer
        model.add(Activation('sigmoid'))  # Removed Activation Layer

        model.compile(loss='mean_absolute_error', optimizer=optimizers.Nadam(), metrics=['accuracy'])
        print(model.summary())        
    
        fitting_res = model.fit(train_X, train_y, epochs=n_epochs, validation_data=(test_X, test_y), verbose=2, shuffle=True)  # train_on_batch()
        end_time = time.time()  # STOP: Training-Time Tracker
        embeds_gen_time = end_time - start_time
        
        # TRAINING: Evaluate model's performance (OVERFITTING = Train LOSS < Test LOSS)
        scores_train = model.evaluate(train_X, train_y, verbose=0)
        print("\nEmbeddings Training:- Mean Abs. Error: %.2f; Accuracy: %.2f%%" % (scores_train[0], scores_train[1]*100))
        
        # VALIDATION: Evaluate model's performance (OVERFITTING = Train MAE < Test MAE)
        scores_validtn = model.evaluate(test_X, test_y, verbose=0)
        print("\nEmbeddings Validation:- Mean Abs. Error: %.2f; Accuracy: %.2f%%" % (scores_validtn[0], scores_validtn[1]*100))
        
        # Accessing the embedding layer through a constructed model 
        # Firstly, `0` refers to the position of embedding layer in the `model`
        # ONLY layers (Dense and/or Embedding) defined before the Flatten() are reported/documented
        # `layer weights` == model.layers[0].get_weights()[0] || `bias weights` == model.layers[0].get_weights()[1]
        # `layer-1 weights` == model.layers[0].get_weights()[0] || `layer-2 weights` == model.layers[1].get_weights()[0] || `layer-3 weights` == model.layers[2].get_weights()[0]
        embeddings = model.layers[0].get_weights()[0]
        
        # `embeddings` has a shape of (num_vocab/input_enc_dim, embedding_dim/output_dim) 
        print("Original Embeddings Shape: ", embeddings.shape)
        embeds = pd.concat([pd.DataFrame(nodes), pd.DataFrame(embeddings)], axis='columns', ignore_index=True)
        embeds.to_csv(fname+'.embeds', sep='\t', header=False, index=False)
        
    return embeds, embeds_gen_time


def inference_predictor(train_X, train_y, test_X, test_y, fname):
    # X = ['source', 'destn'] + col_name
    # y = ['y_cls', 'comn_dist', 'y_reg', 'kendall', 'euclidean_norm', 'cosine_sim']
    X = np.append(train_X, test_X, axis=0)
    y = np.append(train_y, test_y, axis=0)
    
    # Training (Logistic Classifier)
    start_time = time.time()  # START: Training-Time Tracker
    log_clf = LogisticRegression(solver='lbfgs', random_state=42)
    
    log_clf.fit(train_X[:,2:], train_y[:,0])
    
    # Training (Heuristics): Compute class scores
    false_ties_score = list()
    true_ties_score = list()
    unlink_ties_score = list()
    for i in range(X.shape[0]):
        if y[i,0] == 0:
            false_ties_score.append(y[i,2])
        elif y[i,0] == 1:
            true_ties_score.append(y[i,2])
    unlink_ties_score = list(set(false_ties_score).intersection(true_ties_score))
    if len(unlink_ties_score) == 0:
        unlink_ties_score = [None]
    end_time = time.time()  # STOP: Training-Time Tracker
    train_time = end_time - start_time
    print("\nTraining Time: ", train_time, "seconds")
    
    # Save Inference/Deduction scores
    false_ties_score.sort()
    true_ties_score.sort()
    unlink_ties_score.sort()
    inf_scores = pd.concat([pd.DataFrame(false_ties_score), pd.DataFrame(true_ties_score), pd.DataFrame(unlink_ties_score)], axis='columns', ignore_index=True)
    inf_scores.columns = ['false_ties_score', 'true_ties_score', 'unlink_ties_score']
    inf_scores.to_csv(fname+'_deduction.csv', sep=',', header=True, index=False)
    print("\nInference/Deduction scores:")
    print("-----------------------------")
    print(inf_scores)
    
    # Testing (Logistic Classifier)
    preds = log_clf.predict(test_X[:,2:])
    pred_y = np.rint(preds).astype(np.int32)
    
    # Testing (Heuristics): Inference/Deduction
    ties = list()
    unlink = list()
    for j in range(X.shape[0]):
        key = y[j,2]
        if key in unlink_ties_score:
            if false_ties_score.count(key) > true_ties_score.count(key):
                unlink.append(-1)
            else:
                unlink.append(0)
        else:
            unlink.append(0)
        ties.append([X[j,0], X[j,1], key])
        
    # Results: Results of Testing/Validation
    print("'ties' shape: "+str(np.array(ties).shape))
    print("'unlink' shape: "+str(np.array(unlink).shape))
    pred_y_temp = np.append(train_y[:,0], pred_y, axis=0)
    result = pd.concat([pd.DataFrame(ties), pd.DataFrame(y[:,0]), pd.DataFrame(pred_y_temp), pd.DataFrame(unlink)], axis='columns', ignore_index=True)
    result.columns = ['source', 'destn', 'y_reg', 'truth_y_cls', 'pred_y_cls', 'unlink']
    result.to_csv(fname+'.pred', sep='\t', header=True, index=False)
    # Generating 'Unlink' records/rows/entities
    unlink_ties = result.query('unlink == -1 & truth_y_cls == 1')
    unlink_ties.to_csv(fname+'.unlink', sep='\t', header=True, index=False)
    print("\nList of Breakup/Rift Ties:")
    print(unlink_ties)
    
    # Evaluations:
    pred_y_proba = log_clf.predict_proba(test_X[:,2:])[:, 1]
    pred_y_2 = np.rint(log_clf.predict(test_X[:,2:])).astype(np.int32)
    pred_y_3 = np.rint(log_clf.predict(test_X[:,2:])).astype(np.int32)
    evals = evaluations(test_y[:,0], pred_y, pred_y_proba, pred_y_2, pred_y_3)
    
    return evals, result, train_time


#################################################################### Program Flow ####################################################################

def main_prog_flow(myCls, args, edge_list):
    ### Generate graph from edge list ###
    # ".iloc[]" returns a Pandas DATAFRAME
    # ".values[]" returns a NUMPY Array wrt dataframes
    edges, nodes = nodelist_edgelist_compute(myCls, args, edge_list)  # edges, nodes = NUMPY(), NUMPY()
    
    ### Initialize model    
    log_key = 'HeuGAT Model: '+edge_list
    log_file = open('eval_log.txt', 'a')
    print("\n\n----"+log_key+"----", file=log_file)
    print("Evaluating: " + edge_list + "'s dataset\n")
    
    ### Pre-training preprocessing ###
    # Compute FalseHood edges/ties
    graph_fname = edge_list + '/' + edge_list
    fname = args.root_path + graph_fname
    soc_graph = igh.Graph.Read(fname+'.edgelist', format='edgelist', directed=True)
    if not os.path.isfile(fname+'.falseTies'):
        temp_x1 = falsehood_ties_gen(soc_graph, nodes, edges)
        temp_x1.to_csv(fname+'.falseTies', sep=' ', header=False, index=False)
        temp_1 = temp_x1.values[:,:]        
    else:
        temp_x1 = myCls.load_data(args.root_path, graph_fname+'.falseTies', sep='\s', header=None, index_col=None, mode='READ')
        temp_1 = temp_x1.values[:,:]
    print("Shape of 'false_ties (temp_1)': ", (temp_1.shape))
    # Process Truth edges/ties
    row1 = np.ones((edges.shape[0],1), dtype=np.int64)
    temp_2 = np.append(edges, row1, axis=1)
    print("Shape of 'truth_ties (temp_2)': ", (temp_2.shape))
    # Combine/Fuse 'FalseHood' and 'Truth' edges/ties
    temp_3 = np.append(temp_1, temp_2, axis=0)    
    print("Shape of 'temp_3 (false_ties & truth_ties)': ", (temp_3.shape))
    
    ### Preserve ratio/percentage of samples per class using efficent data-splitting && data-resampling strategeies
    train_frac = 0.8
    test_frac = round((1 - train_frac), 1)
    X = temp_3[:,0:-1]
    y = temp_3[:,-1]  # 'y_cls' column
    print("Training classifier using {:.2f}% ties/edges...".format(train_frac * 100))
    if not os.path.isfile(fname+'_strat.splits'):
        stratified_data = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, train_size=train_frac, random_state=42)
        for train_index, test_index in stratified_data.split(X, y):
            strat_X_train, strat_y_train = X[train_index], y[train_index]
            strat_X_test, strat_y_test = X[test_index], y[test_index]
            # Preserve 'train' & 'test' stratified-shuffle-splits
            train_test_splits = pd.concat([pd.DataFrame(train_index), pd.DataFrame(test_index)], axis='columns', ignore_index=True)
            train_test_splits.to_csv(fname+'_strat.splits', sep=' ', header=False, index=False)        
    else:
        strat_train_test = myCls.load_data(args.root_path, graph_fname+'_strat.splits', sep='\s', header=None, index_col=None, mode='READ')
        train_index, test_index = strat_train_test.values[:,0], strat_train_test.values[:,-1]  # "values()" method returns a NUMPY array wrt dataframes
        train_index, test_index = train_index[np.logical_not(np.isnan(train_index))], test_index[np.logical_not(np.isnan(test_index))]  # Remove nan values from arrays
        train_index, test_index = train_index.astype('int32'), test_index.astype('int32')
        strat_X_train, strat_y_train = X[train_index], y[train_index]
        strat_X_test, strat_y_test = X[test_index], y[test_index]     
    print("Shape of 'strat_X_train': %s;  Shape of 'strat_y_train': %s" % (strat_X_train.shape, strat_y_train.shape))
    print("Shape of 'strat_X_test': %s;  Shape of 'strat_y_test': %s" % (strat_X_test.shape, strat_y_test.shape))
    ### Preserve ratio/percentage of samples per class using efficent data-splitting && data-resampling strategeies    
    
    ### Embeddings Creation/Generation
    if not os.path.isfile(fname+'.embeds'):
        min_enc = np.amin(X)
        max_enc = np.amax(X)
        input_enc_dim = (max_enc - min_enc) + 1
        embeds, embed_gen_time = embeddings_gen(myCls, strat_X_train, strat_y_train, strat_X_test, strat_y_test, input_enc_dim, nodes, fname)
        print("Embeddings Generation Time: ", embed_gen_time, "seconds") 
        print("Embeddings Generation Time: ", embed_gen_time, "seconds", file=log_file) 
    else:
        embeds = myCls.load_data(args.root_path, graph_fname+'.embeds', sep='\t', header=None, index_col=None, mode='READ')
    print("Shape of 'Embeddings': ", (embeds.shape))
    
    ### Feature Engineering
    if not os.path.isfile(fname+'.feat'):
        base_feats = embeds.values[:,:]  # base_feats[:,0] = 'node_id' || base_feats[:,1:] = 'embeddings'
        node_fts = pd.DataFrame()
        for u in nodes:
            in_deg = soc_graph.degree(u, mode='in')  # column == 1
            out_deg = soc_graph.degree(u, mode='out')  # column == 2
            pagerank = soc_graph.pagerank(u)  # column == 3
            row = [u, in_deg, out_deg, pagerank]
            node_fts = node_fts.append([row], ignore_index=True)
        node_feats = node_fts.values[:,:]  # 'node_feats' and 'base_feats' possess same node_index at column == 0
        
        fuselage = pd.DataFrame()    
        for i in range(X.shape[0]):
            src_feat_idx = np.where(base_feats[:,0] == X[i,0])  # Returns indices of matched rows
            src_feat_idx = np.asscalar(src_feat_idx[0])
            dest_feat_idx = np.where(base_feats[:,0] == X[i,1])  # Returns indices of matched rows
            dest_feat_idx = np.asscalar(dest_feat_idx[0])
            #vertex_conn = soc_graph.vertex_disjoint_paths(temp_3[i,0], temp_3[i,1])  # k-vertex connected graph == |vertex-cut|
            #edge_conn = soc_graph.edge_disjoint_paths(temp_3[i,0], temp_3[i,1])  # k-edge connected graph == |edge-cut|
            X_feats = list(base_feats[src_feat_idx,1:]) + list(base_feats[dest_feat_idx,1:])
            var_x = (node_feats[src_feat_idx,2]/(len(nodes) - 1)) * node_feats[src_feat_idx,3]  # out_deg_centr * pagerank (* embed)
            var_y = (node_feats[dest_feat_idx,1]/(len(nodes) - 1)) * node_feats[dest_feat_idx,3]  # in_deg_centr * pagerank (* embed)
            comn_dist = np.sum(base_feats[src_feat_idx,1:] - base_feats[dest_feat_idx,1:])
            #y_reg = np.sum((var_x * base_feats[src_feat_idx,1:]) + (var_y * base_feats[dest_feat_idx,1:]))  # Sparse dataset (Zachary-Karate)
            y_reg = cosine_sim((var_x * base_feats[src_feat_idx,1:]), (var_y * base_feats[dest_feat_idx,1:]))  # Dense datasets (CiteSeer, Cora, PubMed-Diabetes, etc.)
            kendall, kendall_pval = stats.kendalltau(base_feats[src_feat_idx,1:], base_feats[dest_feat_idx,1:])
            euc_norm = euclidean_norm(base_feats[src_feat_idx,1:], base_feats[dest_feat_idx,1:])
            cos_sim = cosine_sim(base_feats[src_feat_idx,1:], base_feats[dest_feat_idx,1:])
            #row = [X[i,0], X[i,1]] + X_feats + [y[i], round(comn_dist, 7), round(y_reg, 7), round(kendall, 7), round(euc_norm, 7), round(cos_sim, 7)]
            row = [X[i,0], X[i,1]] + X_feats + [y[i], round(comn_dist, 4), round(y_reg, 4), round(kendall, 4), round(euc_norm, 4), round(cos_sim, 4)]
            fuselage = fuselage.append([row], ignore_index=True)
        # Preserve extracted features
        col_name = list()
        for k in range((embeds.shape[1] - 1) * 2):  # base_feats[:,0] = 'node_id' || base_feats[:,1:] = 'embeddings'
            col_name.append('X'+str(k+1))
        fuselage.columns = ['source', 'destn'] + col_name + ['y_cls', 'Common_Distance', 'y_reg', 'Kendall_Tau', 'Euclidean_Distance', 'Cosine_Similarity']
        fuselage.to_csv(fname+'.feat', sep='\t', header=True, index=False)
    else:
        fuselage = myCls.load_data(args.root_path, graph_fname+'.feat', sep='\t', header=0, index_col=None, mode='READ')
    print("Shape of 'fuselage (false_ties & truth_ties)': ", (fuselage.shape))
    ### Feature Engineering  
    
    ### Compute correlation coefficients wrt dataset
    #data = fuselage[ ['y_reg', 'Common_Distance', 'Euclidean_Distance', 'Cosine_Similarity', 'Kendall_Tau', 'y_cls'] ]
    data = fuselage[ ['y_reg', 'Common_Distance', 'Euclidean_Distance', 'Cosine_Similarity', 'Kendall_Tau', 'y_cls'] ]
    myCls.correlation_coefs(data, 'y_cls', fname)
    
    ### Training && Testing
    train_X = fuselage.values[train_index,:-6]
    train_y = fuselage.values[train_index,-6:]
    test_X = fuselage.values[test_index,:-6]
    test_y = fuselage.values[test_index,-6:]
    evals, result, train_time = inference_predictor(train_X, train_y, test_X, test_y, fname)
    print("Training Time: ", train_time, "seconds", file=log_file)
    
    ### Evaluation Report    
    print("\nEvaluation Report:", file=log_file)
    for i, j in evals.items():
        print(i, ": ", j)
        print(i, ": ", j, file=log_file)
    log_file.close()
    
def loop_prog_flow(myCls, args):
    #graph_data = ["CiteSeer", "Cora", "DBLP", "Facebook-Page2Page", "PubMed-Diabetes", "Wiki", "Zachary-Karate"]
    edge_list = os.listdir(args.root_path)

    for i in range(len(edge_list)):
        main_prog_flow(myCls, args, edge_list)
    
def entry_point():    
    pd.set_option('display.max_columns', None)  # Force Pandas() to display any number of columns
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # Force Numpy() to display any number of columns
    
    # Parse arguments from command-line
    args = args_parse_cmd()
    
    # Load & instantiate generic dependency (class file)
    REMOTE_URL = "https://snap.stanford.edu/data/gemsec_deezer_dataset.tar.gz"
    ROOT_PATH = args.root_path
    FILE_NAME = args.edge_list
    depedencyClass = Starter(REMOTE_URL, ROOT_PATH, FILE_NAME)
    
    # Parse program run-mode
    if args.run_mode == 'single':
        main_prog_flow(depedencyClass, args, args.edge_list)
    elif args.run_mode == 'all':
        loop_prog_flow(depedencyClass, args)
        
if __name__ == "__main__":
    entry_point()