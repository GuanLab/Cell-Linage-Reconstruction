#!/usr/bin/python3
#author: @rayezh
import numpy as np
import pandas as pd
from itertools import combinations
import glob, random, re, os, sys, shutil
import argparse
import dendropy
from Bio import Phylo
from dendropy.calculate import treecompare, treemeasure
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor,DistanceMatrix,_Matrix
from sklearn.ensemble import RandomForestRegressor
from tmc_wrapper.triplets_distance import triplets_score
from tree_evaluations import evaluate_normalized_RF_distance, evaluate_triplet_distance, evaluate_triplet_distance_correlation

def seed_randomness(seed=0):
    random.seed(seed) #restate in argv
    np.random.seed(seed)

def tree2matrix(tree):
    """
    Transform dendropy tree object to arbituary distance matrix
    """
    for node in tree.nodes():
        node.edge.length = 1
    pdm = tree.phylogenetic_distance_matrix()
    matrix = {}
    txns = tree.taxon_namespace
    for  txn1 in txns:
        for txn2 in txns:
            if txn1 != txn2:
                matrix.update({(txn1.label,txn2.label):pdm(txn1, txn2)-1})
    return matrix

def calculate_weights_per_position(idx):
    """
    *Determine the weights of every position based on the mutation rate(mutated/unmutated) and transition rate(0/2) in the training set 
    input:
        idx: index of sub1_train.txt file
    
    output:
        weights: a dataframe of weights of cell states at each position.    

    """
    fulldata = []
    for i in idx:
        path = '../recording_train/sub1_train_'+str(i)+'.txt'
        data = pd.read_csv(path, sep = '\t', header = 0, dtype = str)
        for _,row in data.iterrows():
            fulldata.append([float(x) for x in list(row.state)])
    fulldata = np.array(fulldata)
    row,col = fulldata.shape
    w_2 = []
    w_1 = []
    w_0 = []
    for c in range(col):
        pos = fulldata[:,c]
        count_2 = np.count_nonzero((pos == 2))
        count_0 = np.count_nonzero((pos == 0))
        count_1 = np.count_nonzero((pos == 1))
        w_0.append(-count_0/(count_0+count_2))
        w_1.append(0)
        w_2.append(count_2/(count_0+count_2))
    #data = {1:w_2, -1:w_0, 0:w_1}
    data = {1:w_2, -1:w_0, 0:w_1}
    df = pd.DataFrame(data)
    print(df)
    return df

####question:
    """
    1. the mutation rate of each generation is different, how to use it in clustering?
    2. for clustering, the mudation rate should be rooted for each generation
    3. ths, we use the old rule(back to original state '1' if the childen are mutated to either '1' or '2')
     and use the rooted weight matrix for each position for the new presentation of the matrix;
    4. what if we're comparing cells/clusters from different generations?
        we can use a simulated father of this cell based on mutation rates?
    """


def train_test_split(k = 4):
    import os
    from sklearn.model_selection import ShuffleSplit
    recording_filepaths = glob.glob('../recording_train/sub1_train*')
    groundTruth_filepaths = glob.glob('../groundTruth_train/sub1_train*')
    if len(recording_filepaths) == len(groundTruth_filepaths):
        total = len(recording_filepaths)
        print('Spliting training and test sets from the total of '+str(total)+' datasets ...')
        idx = list(range(1, total+1))
        fold = round(total/k)
        random.shuffle(idx)
        for i in range(1, k+1):
            cut_start = (i-1)*fold
            cut_end = i*fold
            test_idx = idx[cut_start:cut_end] 
            train_idx = [x for x in idx if x not in test_idx]
            print('Start the fold '+str(i)+' in k-fold cross-validation ...')
            print('trainsize:'+str(len(train_idx)))
            print('testsize:'+str(len(test_idx)))
            yield i, train_idx, test_idx
    else:
        print('The train and test sizes mismatch!')
        
def pairwise_sequence(X,Y):
    '''
    input:
        X: recordings of all cells in one cluster in pandas DataFrame format;
           contains two column: 'cell' and 'state'
        Y: groundtruth(tree) of the cell cluster
    output:
        permutated pairwise combinitions of all cells and distances
        X_out: a list of 2 by 10 matrix of the recordings from two cells 
        Y_out: a list of distances between two cells
    '''
    for i in Y.edges():
        i.length = 1/len(Y.edges())
    pdm = treemeasure.PatristicDistanceMatrix(Y)
    X_out = []
    Y_out = []
    for i, tx1 in enumerate(Y.taxon_namespace):
        for j, tx2 in enumerate(Y.taxon_namespace):
            Y_pair = pdm(tx1, tx2)
            state1 =  np.array([float(a) for  a in list(tx1.label.split(' ')[1])])
            state2 =  np.array([float(a) for  a in list(tx2.label.split(' ')[1])])
            #X_pair = np.vstack((state1, state2)) # random forest only takes 1-d feature array
            X_pair = np.hstack((state1, state2)) 
            X_out.append(X_pair)
            Y_out.append(Y_pair)
            #print(X_pair, Y_pair)

    return X_out, Y_out

def UPGMA_tree_reconstruction(dm):
    '''
    input:
        dm: distance matrix
    
    output:
        tree: reconstructed tree
    '''
    constructor = DistanceTreeConstructor()
    tree = constructor.upgma(dm)
    #tree = constructor.nj(dm)
    print(tree)
    return(tree)

def hierachical_clustering(X, weights):
    '''
    input: recordings of cells
    output: clustered tree
    '''
    #weights = np.array([2,2,2,2,2,2,2,2,2,2])
    lineage = {str(row.cell)+' '+str(row.state):np.array([(float(x)-1) for x in list(row.state)]) for _, row in X.iterrows()}
    #lineage = {str(row.cell)+' '+str(row.state):np.array([weights[x][i] for i,x in enumerate(list(row.state))]) for _, row in X.iterrows()}
    txns = dendropy.TaxonNamespace([str(row.cell)+' '+str(row.state) for _, row in X.iterrows()])
    nodes = {txn.label:dendropy.Node(edge_length = 1, label = txn.label, taxon =txn) for txn in txns}
    while len(lineage) >= 2:
        #find the nearest two cells
        #compute distance matrix and find minimum
        #dist = [[sum(abs(linage[i] - linage[j])) for i in linage.keys()] for j in linage.keys()]
        diff = 1000000
        # the next parts are for the duplicated elements in cell states
        # unique elements of lineage values(np.ndarrays)
        #uniq_lineage_val = []
        #for val in lineage.values():
        #    if not any(np.array_equal(val, uniq) for uniq in uniq_lineage_val):
        #        uniq_lineage_val.append(val)
        #for taxa in uniq_lineage_val:
        #    keys = [key for key,value in lineage.items() if np.array_equal(value,taxa)]
        #    if len(keys)>1:
        #        cluster = lineage[keys[0]]
        #        children = []
        #        cluster_name= ''
        #        for key in keys:
        #            cluster_name=cluster_name+key+' '
        #            lineage.pop(key)
        #            children.append(nodes[key])
        #            nodes.pop(key)
        #        cluster_name = cluster_name[:-1]
        #        lineage.update({cluster_name:cluster})
        #        ch_new = dendropy.Node(edge_length = 1)
        #        ch_new.set_child_nodes(children)
        #        nodes.update({cluster_name:ch_new})
        pairs = [pair for pair in  combinations(lineage.keys(),2)]
        random.shuffle(pairs)
        for pair in pairs:
            weighted_i = np.array([weights[state][pos] for pos, state in enumerate(lineage[pair[0]])])
            weighted_j = np.array([weights[state][pos] for pos, state in enumerate(lineage[pair[1]])])
            diff_new = sum(abs(weighted_i - weighted_j))
            if (diff>=diff_new):
                #print(weighted_i, lineage[i])
                #diff = sum(abs(lineage[i] - lineage[j]))
                diff = diff_new
                sister = pair
        #create new cluster
        #print((sister),diff,lineage[sister[0]],lineage[sister[1]])
        cluster_new = {sister[0]+' '+sister[1]:np.round((lineage[sister[0]]+lineage[sister[1]])/2)+0.0}
        #cluster_new = {sister[0]+' '+sister[1]:(lineage[sister[0]]+lineage[sister[1]])/2+0.0}
        #delete two cells from linage and add new cluster
        #print(lineage[sister[0]])
        #print(lineage[sister[1]])
        #print(cluster_new)
        lineage.pop(sister[0])
        lineage.pop(sister[1])
        lineage.update(cluster_new)
        #add nodes:
        #print(sister[0])
        ch_0 = nodes[sister[0]]
        ch_1 = nodes[sister[1]]
        ch_new = dendropy.Node(edge_length = 1)
        ch_new.set_child_nodes([ch_0, ch_1])
        nodes.pop(sister[0])
        nodes.pop(sister[1])
        nodes.update({sister[0]+' '+sister[1]:ch_new})
    
    root = list(nodes.values())[0]
    root.label = "root"
    tree = dendropy.Tree(taxon_namespace=txns, seed_node = root)
    #tree.print_plot()
    return tree



def read_training_pairwise(idx):
    '''
    read pairwised recordings and groundtruth(tree) for train/test sets from indexes 
    '''
    X_pairs_all = []
    Y_pairs_all = []
    for i in idx:
        X = pd.read_csv('../recording_train/sub1_train_'+str(i)+'.txt', sep = '\t', header = 0, dtype = str)
        Y = dendropy.Tree.get_from_path('../groundTruth_train/sub1_train_'+str(i)+'.nwk', 'newick')
        X_pairs, Y_pairs = pairwise_sequence(X,Y)
        X_pairs_all.extend(X_pairs)
        Y_pairs_all.extend(Y_pairs)
    X_pairs_all = np.array(X_pairs_all)
    Y_pairs_all = np.array(Y_pairs_all)
    print(X_pairs_all.shape, Y_pairs_all.shape)
    return X_pairs_all,Y_pairs_all

def random_forest_learner(X_train, Y_train):
    regressor = RandomForestRegressor(n_estimators=500, random_state=0)
    regressor.fit(X_train, Y_train)
    return regressor
    #Y_pred = regressor.predict(X_test)
    #mse = ((Y_test - Y_pred)**2).mean(axis = 0) 
    #print(mse)
    #a = np.vstack((Y_test, Y_pred))
    #ia, ja = a.shape
    #for j in range(ja):
    #    print(a[:,j])


def test(idx, regressor):
    '''
    '''
    predfile = open('prediction.txt', 'w')
    predfile.write("\"dreamID\"\t\"nCells\"\t\"RF\"\t\"ground\"\t\"rec\"\n")
    for i in idx:
        X = pd.read_csv('../recording_train/sub1_train_'+str(i)+'.txt', sep = '\t', header = 0, dtype = str)
        Y = dendropy.Tree.get_from_path('../groundTruth_train/sub1_train_'+str(i)+'.nwk', 'newick')
        names = [str(row.cell)+' '+str(row.state) for _, row in X.iterrows()]
        matrix = []
        # the diagnal of distance matrix should all be 0
        for i_1, row_1 in X.iterrows():
            dist = []
            state_1 = np.array([float(x) for x in list(row_1.state)])
            for i_2, row_2 in X.iterrows():
                if i_2 < i_1:
                    state_2 = np.array([float(x) for x in list(row_2.state)])
                    x_pair = np.hstack((state_1, state_2))
                    y_pred = regressor.predict(x_pair.reshape(1,-1))[0]
                    dist.append(y_pred)
                elif i_2 == i_1:
                    y_pred = 0
                    dist.append(y_pred)
            matrix.append(dist)
        #print(matrix) 
        #print(i,names)
        m = DistanceMatrix(names, matrix)
        rec_tree = UPGMA_tree_reconstruction(m)
        Phylo.write(rec_tree, 'sub1_train_'+str(i)+'.nwk', 'newick')
        rec_tree = dendropy.Tree.get(path = 'sub1_train_'+str(i)+'.nwk', schema = 'newick',taxon_namespace = Y.taxon_namespace)
        evaluation = evaluate_normalized_RF_distance(Y, rec_tree)
        print("dreamID: %d; RF: %.6f" % (i, evaluation))
        predfile.write("%d\t%d\t%.15f\t%s\t%s\n" % (i, len(Y), evaluation, Y.as_string(schema = 'newick').rstrip(), rec_tree.as_string(schema = 'newick').rstrip()))
    predfile.close()

def main_1():
    #parser = argparse.ArgumentParser(description='Build Phylogenetic Tree.')
    #parser.add_argument('--tree_method', default = 'upgma', dest = 'tree_method', help='tree construction method, UPGMA(upgma) or Neighbor Joining(nj)')
    k = 4
    seed = 0
    seed_randomness(seed)
    for i, train_idx, test_idx in train_test_split(k):
        path = './fold'+str(i)
        if not os.path.exists(path):
            os.mkdir(path, 0o755)
        X_train,Y_train =read_training_pairwise(train_idx)
        regressor = random_forest_learner(X_train, Y_train)
        test(test_idx, regressor)
        for filepath in glob.glob('./*.nwk'):
            shutil.move(filepath, path)
        for filepath in glob.glob('./*.txt'):
            shutil.move(filepath, path)



def main_2():
    #parser = argparse.ArgumentParser(description='Build Phylogenetic Tree.')
    #parser.add_argument('--tree_method', default = 'upgma', dest = 'tree_method', help='tree construction method, UPGMA(upgma) or Neighbor Joining(nj)')
    k = 4
    seed = 0
    seed_randomness(seed)
    ################# read train/test indx from yuanfang's file ############
    train_test_split = []
    #for i in range(10):
    for i in ['all']:
        train_idx = []
        #with open('train_gs_'+str(i)+'.dat','r') as trainfile:
        with open('train_gs_'+str(i)+'.dat','r') as trainfile:
        #print(trainfile.read())
            for line in trainfile:
                table = line.rstrip().split('.nwk')
                table = table[0]
                table = table.split('_')
                table = int(table[-1])
                #print(table)
                train_idx.append(table)
        test_idx = []
        with open('train_gs_'+str(i)+'.dat','r') as testfile:
            for line in testfile:
                table = line.rstrip().split('.nwk')
                table = table[0]
                table = table.split('_')
                table = int(table[-1])
                test_idx.append(table)
        train_test_split.append((i,train_idx, test_idx))
        #print(train_test_split)
    #for i, train_idx, test_idx in train_test_split(k):
    for i,train_idx, test_idx  in train_test_split:
        path = './fold'+str(i)
        if not os.path.exists(path):
            os.mkdir(path, 0o755)
        predfile = open('prediction.txt', 'w')
        predfile.write("\"dreamID\"\t\"nCells\"\t\"RF\"\t\"Triplet\"\t\"ground\"\t\"rec\"\n")
        RF_score = 0
        triplet_score = 0
        RF_j = 0
        triplet_j = 0
        weights = calculate_weights_per_position(train_idx)
        for j in test_idx:
            X = pd.read_csv('../recording_train/sub1_train_'+str(j)+'.txt', sep = '\t', header = 0, dtype = str)
            Y = dendropy.Tree.get_from_path('../groundTruth_train/sub1_train_'+str(j)+'.nwk', 'newick')
            #Y.is_rooted = True
            rec_tree = hierachical_clustering(X, weights)
            rec_tree.write(path='sub1_train_'+str(j)+'.nwk',schema='newick')
            rec_tree = dendropy.Tree.get(path = 'sub1_train_'+str(j)+'.nwk', schema = 'newick',taxon_namespace = Y.taxon_namespace)
            #rec_tree.is_rooted = True
            RF = evaluate_normalized_RF_distance(Y, rec_tree)
            #triplet = evaluate_triplet_distance(Y, rec_tree)
            #triplet = evaluate_triplet_distance_optimized(Y, rec_tree)
            counts = triplets_score(rec_tree, Y, n=1000)
            triplet = counts[True]/(counts[True]+counts[False])
            #print(counts[True]+counts[False])
            if not np.isnan(RF):
                RF_score = RF_score+RF
                RF_j+=1
                #print(score, j)
            triplet_score = triplet_score+triplet
            triplet_j +=1
            print("dreamID: %d; RF: %.6f; Triplet: %.6f" % (j, RF, triplet))
            #print(rec_tree.as_string(schema = 'newick').rstrip())
            # reformat the tree
            predfile.write("%d\t%d\t%.15f\t%.15f\t\"%s\"\t\"%s\"\n" % (j, len(Y), RF,triplet, Y.as_string(schema = 'newick', suppress_rooting=True).rstrip(), rec_tree.as_string(schema = 'newick', suppress_rooting=True).rstrip()))
        predfile.close()
        print(RF_score/RF_j, triplet_score/triplet_j)
        for filepath in glob.glob('./*.nwk'):
            shutil.move(filepath, path)
        for filepath in glob.glob('./*.txt'):
            shutil.move(filepath, path)

if __name__ == '__main__':
    main_2()
