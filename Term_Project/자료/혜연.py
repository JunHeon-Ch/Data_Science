import pandas as pd
import numpy as np
import math
import json
import random
import graphviz
import pydot
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

original_file_name = "C:\\Users\\CHOI\\Desktop\\Dataset-Unicauca-Version2-87Atts.csv"
dirty_file_name = "C:\\Users\\CHOI\\Desktop\\D.csv"
ReadingCount = 100000;


def debug():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


data = pd.read_csv(original_file_name, nrows=ReadingCount)


def ChangeCategory(name, new_name, init, func=None):
    data[name] = data[name].astype("category")

    for i in list(data[name].cat.categories):
        count = len(data[data[name] == i])
        if not i in init:
            if (func == None):
                init[i] = i
            else:
                init[i] = func(i, count)

    data[new_name] = data[name].apply(lambda x: init[x])


def RemoveColumn(name):
    global data
    data = data.loc[:, data.columns != name]


def CreateMissing():
    dataSize = data.shape[0]
    colSize = data.shape[1]

    for i in range(int(dataSize * colSize * 0.0012)):
        data.iat[random.randrange(dataSize), random.randrange(colSize)] = None


def CreateWrong(name, select1, select2, select3):
    datap = data.loc[data[name] == select1]
    temp = []
    for i in range(0, 3):
        select = datap.sample(frac=0.5, replace=False)
        datap = datap.drop(select.index)
        temp.append(select.index)
    data.loc[temp[0], name] = select1
    data.loc[temp[1], name] = select2
    data.loc[temp[2], name] = select3


def CreateUnusable(name):
    data[name] = data[name].apply(str)
    datap = data
    temp = []
    for i in range(0, 2):
        select = datap.sample(frac=0.5, replace=False)
        datap = datap.drop(select.index)
        temp.append(select.index)

    select_no = 0
    data.loc[temp[select_no], name] = ((data.loc[temp[select_no], name]).astype("float") / 1024).apply(str) + " KB"
    select_no = 1
    data.loc[temp[select_no], name] = ((data.loc[temp[select_no], name]).astype("float") / 1024 / 1024).apply(
        str) + " MB"


def DeleteUnusable(name):
    if (str(data[name].dtype) == "object"):
        count = 0
        kb = data[name].str.upper().str.contains("KB")
        temp = ((data.loc[kb, name].str.extract('([\d]+[.][\d]+)').astype(float)) * 1024).astype("str")[0];
        count += len(temp)
        data.loc[kb, name] = temp

        mb = data[name].str.upper().str.contains("MB")
        temp = ((data.loc[mb, name].str.extract('([\d]+[.][\d]+)').astype(float)) * 1024 * 1024).astype("str")[0];
        count += len(temp)
        data.loc[mb, name] = temp
        if (count > 0):
            data[name] = data[name].astype(float)


def OneHot(name):
    category = list(data[name].astype("category").cat.categories)
    tcpudp = pd.get_dummies(data[name])

    for i in category:
        n_nae = "category_" + name + "_" + str(i)
        data[n_nae] = tcpudp[i]

    RemoveColumn(name)


CreateUnusable("Total.Length.of.Fwd.Packets")
CreateUnusable("Total.Length.of.Bwd.Packets")
CreateUnusable("Fwd.Packet.Length.Max")
CreateUnusable("Fwd.Packet.Length.Mean")
CreateUnusable("Fwd.Packet.Length.Std")
CreateUnusable("Bwd.Packet.Length.Max")
CreateUnusable("Bwd.Packet.Length.Std")
CreateUnusable("Bwd.Packet.Length.Mean")
CreateUnusable("Fwd.Header.Length")
CreateUnusable("Bwd.Header.Length")
CreateUnusable("Max.Packet.Length")
CreateUnusable("Packet.Length.Mean")
CreateUnusable("Packet.Length.Std")
CreateUnusable("Packet.Length.Variance")
CreateUnusable("Fwd.Header.Length.1")

CreateWrong("ProtocolName", "GOOGLE", "google", "Gogle")
CreateWrong("ProtocolName", "FACEBOOK", "facebook", "fecebok")
CreateWrong("ProtocolName", "YOUTUBE", "youtube", "yourtube")
CreateWrong("ProtocolName", "MICROSOFT", "MS", "microsoft")

CreateMissing()

# data.to_csv(dirty_file_name)
data = data.dropna()
for i in data:
    DeleteUnusable(i)
for i in data:
    if (i == "Protocol"):
        continue;
    if (data[i].dtype == np.float):
        a = np.max(data[i]) - np.min(data[i])
        if (a == 0):
            RemoveColumn(i)
        else:
            data[i] = (data[i] - np.mean(data[i])) / (np.max(data[i]) - np.min(data[i])) * 100

debug()
print(data["ProtocolName"])
debug()
data["ProtocolName"] = data["ProtocolName"].str.upper()
data.loc[data["ProtocolName"] == "GOGLE", "ProtocolName"] = "GOOGLE"
data.loc[data["ProtocolName"] == "YOURTUBE", "ProtocolName"] = "YOUTUBE"
data.loc[data["ProtocolName"] == "MS", "ProtocolName"] = "MICROSOFT"
data.loc[data["ProtocolName"] == "fecebok", "ProtocolName"] = "FACEBOOK"
print(data["ProtocolName"])
RemoveColumn("Flow.ID")
RemoveColumn("Source.IP")
RemoveColumn("Source.Port")
RemoveColumn("Destination.IP")
RemoveColumn("Timestamp")
RemoveColumn("Label")
RemoveColumn("L7Protocol")


def cage_port(i, count):
    if (i >= 49152):
        return "Dynamic Port"
    elif (count > 30):
        return str(i)
    else:
        return "Other Registered Port"



ChangeCategory("Destination.Port", "Destination.Port", {
    21: "FTP",
    80: "HTTP",
    2221: "Common Transformation FTP",
    3128: "Proxy",
    443: "HTTPS"
}, cage_port)

ChangeCategory("Protocol", "Protocol", {
    17: "UDP",
    6: "TCP"
})

OneHot("Protocol")
OneHot("Destination.Port")

# Shuffle the data
temp = data.sample(frac=1)
# Divide Set
size = int(len(temp) / 5);
subsets = []
for i in range(0, 5):
    subsets.append(temp[size * i:size * (i + 1)])

k_fold_probability = []
# Get each test_set and training_set
for test_set in subsets:
    count = 0
    training_set = temp.drop(test_set.index)

    x = training_set.loc[:, training_set.columns != "ProtocolName"]
    y = training_set["ProtocolName"]
    ree1 = DecisionTreeClassifier(criterion='entropy', max_depth=50, random_state=200).fit(x, y)

    test_y = list(test_set["ProtocolName"])
    test_predict_y = ree1.predict(test_set.loc[:, test_set.columns != "ProtocolName"])

    count = 0
    for i in range(len(test_y)):
        if (test_y[i] == test_predict_y[i]):
            count += 1;

    k_fold_probability.append(count / len(test_set))

print(np.mean(k_fold_probability))

training_set = data.sample(frac=4 / 5, replace=False)
test_set = data.drop(training_set.index)

x = training_set.loc[:, training_set.columns != "ProtocolName"]
y = training_set["ProtocolName"]
ree1 = DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=200).fit(x, y)

test_y = list(test_set["ProtocolName"])
test_predict_y = ree1.predict(test_set.loc[:, test_set.columns != "ProtocolName"])

# print(confusion_matrix(test_y, test_predict_y))
print(classification_report(test_y, test_predict_y))