import pandas as pd
import dendropy
from train import evaluate_normalized_RF_distance
from dendropy.calculate import treecompare, treemeasure

a = pd.read_csv('train_setDREAM2019.txt', header = 0, sep = '\t')
for i, row in a.iterrows():
    tree1 = dendropy.Tree.get(string = row.ground, schema = 'newick')
    #tree1.is_rooted = True
    tree2 = dendropy.Tree.get(string = row.rec, schema = 'newick', taxon_namespace = tree1.taxon_namespace)
    #tree2.is_rooted = True
    eva = evaluate_normalized_RF_distance(tree1, tree2)
    print(eva)

