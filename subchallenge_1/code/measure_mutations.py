import dendropy
import pandas as pd
import numpy as np
import glob
paths = glob.glob('../recording_train/sub1_train*.txt')
fulldata = []
for filepath in paths:
    data = pd.read_csv(filepath, sep = '\t', header = 0, dtype = str)
    for _,row in data.iterrows():
        #print(row)
        fulldata.append([float(x) for x in list(row.state)])
fulldata = np.array(fulldata)
#print(fulldata)
row,col = fulldata.shape
result = open('statistics.txt', 'w')
for c in range(col):
    print('position',c,':')
    pos = fulldata[:,c]
    count_2=np.count_nonzero((pos == 2))
    count_0=np.count_nonzero((pos == 0))
    count_1=np.count_nonzero((pos == 1))
    print("2:",count_2)
    print("0:",count_0)
    print("1:",count_1)
    result.write('position\t2\t0\t1\n')
    result.write('%d\t%d\t%d\t%d\n' % (c, count_2, count_0, count_1))

result.close()

