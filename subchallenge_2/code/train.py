#!/usr/bin/python4
#author: @rayezh
import numpy as np
import pandas as pd
from itertools import combinations
import glob, random, re, os, sys, shutil
import argparse
import dendropy
from Bio import Phylo
import math
from nexus import NexusReader
from dendropy.calculate import treecompare, treemeasure
from tree_evaluations import evaluate_normalized_RF_distance,evaluate_triplet_distance, evaluate_triplet_distance_correlation 
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor,DistanceMatrix,_Matrix
from sklearn.ensemble import RandomForestRegressor
from tmc_wrapper.triplets_distance import triplets_score
from collections import Counter, defaultdict
import time

def one_hot_embedding(lineage):
    """
    embedding: 
    symbol_set[loc][mutation] = weight
    """
    import string
    from collections import defaultdict
    import numpy as np
    all_symbol = []
    symbol_set =  {x:defaultdict(lambda:0) for x in range(1000)}
    for _,cell in lineage.items():
        for i in range(len(cell)):
            symbol_set[i][cell[i]] +=1
        all_symbol.extend(cell) 
    all_symbol = sorted(list(set(all_symbol)))
    for k1, v1 in symbol_set.items():
        for k2, v2 in v1.items():
            symbol_set[k1][k2] = np.log10(100/v2)+1  #log base =100
    return symbol_set, all_symbol

        
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

def fix_gap(record):
    """
    mark every gap with start and end; gap can be uniquely markded by mutations at both ends: mutation type and location 
    mark tri-gap and more
    """
    i = 0
    gap = False
    #print(record)
    record_new = record.copy()
    while i < len(record):
        n = 0
        #print(record[i], gap)
        if(record[i] =='-')and(gap == False)and(record[i-1] in list('0'+'ABCDEFGHIJKLMNOPQRSTUVWXYZ'+'abcd')):
            start = record[i-1]
            start_idx = i-1
            gap = True
            #score = embed[start]
            sign = 'gap'+start+str(start_idx)
        if(record[i] !='-')and(gap==True)and(record[i] in list('0'+'ABCDEFGHIJKLMNOPQRSTUVWXYZ'+'abcd')):
            end = record[i]
            end_idx = i
            #score = score*embed[end] #5 is arbitury
            sign = sign+'_'+end+str(end_idx)
            if (i+1 == len(record))or (record[i+1] != '-'): # gap ends
                gap = False
                for j in range(start_idx, end_idx+1):
                    record_new[j]=sign
        i+=1
    return record_new#, embed

def find_nearest_pair(lineage, embed,  mutation_all):
    """
    input:
    lineage: a dictionary of taxa:record ['A', 'B', 'C', ...]
    output: 
    sister:the taxa of nearest pair
    """
    from scipy.sparse import csr_matrix
    #print('finding the nearest pair in',len(lineage),'cells ...')
    num_mutation_all =len(mutation_all)
    #print(lineage.values())
    #n = len(lineage[0]) #200
    n = 1000
    mat = np.zeros((len(lineage), num_mutation_all*n))  #cell mutation  matrix for 100 cells; every cell has 200 barcodes
    dist_mat = np.zeros((len(lineage), num_mutation_all*n))
    i = 0
    for k, v in lineage.items():
        for idx,state in enumerate(v):
            if state =='0':
                mat[i, idx*num_mutation_all:(idx+1)*num_mutation_all] = 1
            else:
                mat[i, idx*num_mutation_all+mutation_all.index(state)] = num_mutation_all*embed[idx][state] #num_mutation_all to lift the weights
            
            if state.startswith('gap'):
                dist_mat[i, idx*num_mutation_all:(idx+1)*num_mutation_all] = 0
            else:
                dist_mat[i, idx*num_mutation_all:(idx+1)*num_mutation_all] = 1
        i+=1

    product = mat.dot(mat.T)

    dist_product = dist_mat.dot(dist_mat.T)
    product = np.divide(product, dist_product)
    np.fill_diagonal(product, 0)
    x = len(lineage.keys())
    idx = product.argmax()
    while (idx//x) == (idx % x):
        product[(idx//x), (idx//x)] -= 1
        idx = product.argmax()
    
    pair1 = list(lineage.keys())[(idx//x)]
    pair2 = list(lineage.keys())[(idx % x)]
    print(product[(idx//x), (idx % x)],(pair1, pair2))
    return[pair1, pair2]


def hierachical_clustering(X):
    '''
    input: recordings of cells
    output: clustered tree
    '''
    lineage = {}
    for taxa, record in X:
        record = fix_gap(record)
        lineage.update({taxa:record})
    #embed, all_symbol = one_hot_embedding(lineage)   #customed weights
    while len(lineage) >= 2:
        embed, all_symbol = one_hot_embedding(lineage)
        #find the nearest two cells based on
        #compute distance matrix and find minimum
        sister = find_nearest_pair(lineage, embed, all_symbol)      
        record_new = []
        for i in range(len(lineage[sister[0]])):
            if lineage[sister[0]][i]==lineage[sister[1]][i]:
                record_new.append(lineage[sister[0]][i])
            else:
                if lineage[sister[0]][i].startswith('gap') and (not lineage[sister[1]][i].startswith('gap')):
                    record_new.append(lineage[sister[1]][i])
                elif lineage[sister[1]][i].startswith('gap') and (not lineage[sister[0]][i].startswith('gap')):
                    record_new.append(lineage[sister[0]][i])
                else:
                    record_new.append('0')
        cluster_new = {'('+sister[0]+','+sister[1]+')':record_new}
        lineage.pop(sister[0])
        lineage.pop(sister[1])
        lineage.update(cluster_new)
        #add nodes:
        
    rec_tree = list(lineage.keys())[0]+'root;'
    #print(rec_tree)
    return rec_tree



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

def main():
    #parser = argparse.ArgumentParser(description='Build Phylogenetic Tree.')
    #parser.add_argument('--tree_method', default = 'upgma', dest = 'tree_method', help='tree construction method, UPGMA(upgma) or Neighbor Joining(nj)')
    #for i, train_idx, test_idx in train_test_split(k):
    #    path = './fold'+str(i)
    #    if not os.path.exists(path):
    #        os.mkdir(path, 0o755)
    #    X_train,Y_train =read_training_pairwise(train_idx)
    #    regressor = random_forest_learner(X_train, Y_train)
    #    test(test_idx, regressor)
    #    for filepath in glob.glob('./*.nwk'):
    #        shutil.move(filepath, path)
    #    for filepath in glob.glob('./*.txt'):
    #        shutil.move(filepath, path)
    X = []
    with  open('../data/C3_test.tsv', 'r') as Xfile:
        i = 0
        for line in Xfile:
            if i > 0:
                table = line.rstrip().split('\t')
                X.append((table[0], table[1:]))
                #print(len(table[1:]))
            i += 1
        #Y = dendropy.Tree.get_from_path('../SubC2_train_REF/SubC2_train_'+format(i,'04d')+'_REF.nw', 'newick')
    rec_tree = hierachical_clustering(X)
    rec_file = open('C3_test.nw', 'w')
    rec_file.write(rec_tree)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
