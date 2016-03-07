import csv
import os
import sys

script, first = sys.argv

PATH_UCR = '../UCR_TS_Archive_2015/'

model = first

tabDataset = os.listdir(PATH_UCR)

csvUCR = open('./UCR_Data_results.csv', 'rb')
fileUCR = csv.reader(csvUCR, delimiter=',')          # Contain the list of result of all datasets
tabFileUCR = []
for row in fileUCR:
    tabFileUCR.append(row)
csvUCR.close()

header = tabFileUCR[0]
indexModelErrorRate = header.index(model)

for i in tabDataset:
    try:
        with open(PATH_UCR + i + '/' + i + '_result', 'rb') as csvDataset:
            fileResult = csv.reader(csvDataset, delimiter=',')          # Contain the list of result of one dataset
            headerResult = fileResult.next()
            indexErrorRate = headerResult.index('error rate')
            indexModel = headerResult.index('model')
            tabErrorRate = []
            for j in fileResult:
                if j[indexModel] == model:
                    tabErrorRate.append(j[indexErrorRate])
            tabErrorRate.sort()
            for j in tabFileUCR:
                if j[0] == i:
                    print j[0]
                    j[indexModelErrorRate] = round(float(tabErrorRate[0]), 3)
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror) + " :\t" + i

csvUCR = open('./UCR_Data_results.csv', 'wb')
writerFileUCR = csv.writer(csvUCR, delimiter=',')          # Contain the list of result of all datasets

for row in tabFileUCR:
    writerFileUCR.writerow(row)

csvUCR.close()


















#
