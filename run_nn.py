import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

DATA_LOCATION = './data/data3node.txt'
LIST_LOCATION = './data/lists_3nodes.txt'
RESULTS_LOCATION = './data/results_3nodes.txt'

scaler = StandardScaler()
model = MLPClassifier()

def preprocess():
    data = []
    results = []

    with open(LIST_LOCATION) as listFile:
        with open(RESULTS_LOCATION) as resFile:
            for lis, res in zip(listFile,resFile):
                intList = list(map(int, lis[:-1].split(',')))

                results.append(int(res[:-1]))
                data.append(intList)

    return pd.DataFrame(data), pd.DataFrame(results)

def getAbsError(data, res):
    preds = model.predict(data)
    tot_dif = 0
    for pred, actl in zip(preds, res):
        tot_dif += abs(pred - actl)
#       print("Pred = %d, Actual = %d, --- total = %d" %(pred, actl, tot_dif))

    tot_dif /= len(data)

    return tot_dif

data, res = preprocess()
dat_trn, dat_tst, res_trn, res_tst = train_test_split(data, res, test_size=0.2, random_state=42)

model.fit(dat_trn, res_trn.iloc[:,0])

#acc_trn = accuracy_score(res_trn.iloc[:,0], model.predict(dat_trn))
#acc_tst = accuracy_score(res_tst.iloc[:,0], model.predict(dat_tst))

#print("Training accuracy is: %.2f" %(acc_trn))
#print("Test accuracy is: %.2f" %(acc_tst))
tot_dif = getAbsError(dat_tst, res_tst.iloc[:,0])
print("Absolute error is %.3f" %(tot_dif))

# Training accuracy: 0.28
# Test accuracy: 0.24
