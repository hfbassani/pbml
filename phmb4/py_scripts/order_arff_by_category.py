import argparse
import pandas as pd
import numpy as np
from scipy.io import arff

def getType (type):
    if type == "numeric":
        return "real"

parser = argparse.ArgumentParser()
parser.add_argument('-f', help='Arff files path', required=True)
args = parser.parse_args()

data, meta = arff.loadarff(open(args.f, 'rb'))
data = pd.DataFrame(data)

orderedFile = open("teste.arff", 'w+')
orderedTrueFile = open("teste.true", 'w+')

orderedTrueFile.write("DIM={0};{1} TRUE\n".format())

labels = data['class'].unique()
orderedData = pd.DataFrame()
for label in labels:
    labelData = data[data['class'].isin([label])]
    orderedData = orderedData.append(labelData, ignore_index=True)


orderedFile.write("@relation {0}\n".format(meta.name))

for i in xrange(len(meta.names())):
    attr = meta.names()[i]
    if attr != "class":
        orderedFile.write("@attribute {0} {1}\n".format(attr, getType(meta.types()[i])))
    else:
        orderedFile.write("@attribute {0} {{".format(attr))
        orderedFile.write("{0}".format(",".join(labels)))
        orderedFile.write("}\n")

orderedFile.write("@data\n")

for index, row in orderedData.iterrows():
    dataRow = list(row)
    i = 0
    orderedFile.write("{0}".format(dataRow[i]))
    i += 1

    while i < len(dataRow) - 1:
        orderedFile.write(",{0}".format(dataRow[i]))
        i += 1

    orderedFile.write(",{0}\n".format(dataRow[i]))


orderedData[orderedData['class'] == '0'].index.tolist()