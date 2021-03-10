#!/usr/bin/python3
#author: @rayezh
import numpy as np
import pandas as pd
from itertools import combinations
import re, os, sys
from glob import glob
import dendropy
from Bio import Phylo
from dendropy.calculate import treecompare, treemeasure
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor,DistanceMatrix,_Matrix

def seed_randomness(seed=0):
    random.seed(seed) #restate in argv
    np.random.seed(seed)

def tree2matrix(tree):
    """ Transform dendropy tree object to arbituary distance matrix
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

def calculate_weights_per_position(fpath):
    """ Determine the weights of every position based on the mutation rate(mutated/unmutated) and transition rate(0/2) in the training set 
    input:
        fpath: list
            path of sub1_train.txt file
    
    output:
        weights: a dataframe of weights of cell states at each position.    

    """
    fulldata = []
    for path in fpath:
        print(path)
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
    lineage = {str(row.cell)+'_'+str(row.state):np.array([(float(x)-1) for x in list(row.state)]) for _, row in X.iterrows()}
    while len(lineage) >= 2:
        #find the nearest two cells
        #compute distance matrix and find minimum
        diff = 1000000
        # the next parts are for the duplicated elements in cell states
        # unique elements of lineage values(np.ndarrays)
        pairs = [pair for pair in  combinations(lineage.keys(),2)]
        #random.shuffle(pairs)
        for pair in pairs:
            weighted_i = np.array([weights[state][pos] for pos, state in enumerate(lineage[pair[0]])])
            weighted_j = np.array([weights[state][pos] for pos, state in enumerate(lineage[pair[1]])])
            diff_new = sum(abs(weighted_i - weighted_j))
            if (diff>=diff_new):
                diff = diff_new
                sister = pair
        #create new cluster
        cluster_new = {'('+sister[0]+','+sister[1]+')':np.round((lineage[sister[0]]+lineage[sister[1]])/2)+0.0}
        #delete two cells from linage and add new cluster
        lineage.pop(sister[0])
        lineage.pop(sister[1])
        lineage.update(cluster_new) #add new node
    tree = list(lineage.keys())[0]+'root;'
    print(tree)
    return tree



def read_training_pairwise(idx):
    ''' Read pairwised recordings and groundtruth(tree) for train/test sets from indexes 
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

def main():
    ################# read train/test indx from yuanfang's file ############
    inputpath = sys.argv[1]
    file_path = glob(inputpath+'/*.txt')#glob('../recording_test/*.txt')
    out_path = './output/'
    os.makedirs(out_path, exist_ok = True)  # store individual .nwk file
    predfile = open('prediction.txt', 'w')
    predfile.write("dreamID\tnw\n")
    weights = calculate_weights_per_position(file_path)
    
    for p in file_path:
        X = pd.read_csv(p, sep = '\t', header = 0, dtype = str)
        rec_tree = hierachical_clustering(X, weights)
        rec_file = open(out_path+p.split('/')[-1]+'.nwk','w')
        rec_file.write(rec_tree)
        rec_file.close()
        # reformat the tree
        predfile.write("%s\t%s\n" % (p.split('/')[-1],  rec_tree))
    predfile.close()

if __name__ == '__main__':
    main()
