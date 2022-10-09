import numpy as np
import pandas as pd
from math import sqrt, pow


def common_data(list1, list2) -> bool:
    """
    This function look for a common element between two arrays and return a boolean\n
    common_data(['a', 'b'], ['c', 'd']) returns False\n
    common_data(['a', 'b'], ['c', 'a']) returns True\n
    """
    result = False
    for x in list1:
        for y in list2:
            if x == y:
                result = True
                return result
    return result


def norme(p1, p2, cols):
    res = 0
    for col in cols:
        res += pow((p1[col] - p2[col]), 2)
    return sqrt(res)


def getCenterDot(playersList, infos_to_calculate):
    means = {}
    len_of_players = len(playersList)
    for info in infos_to_calculate:
        sum = 0
        for player in playersList:
            sum += player[info].max()
        means[info] = (sum/len_of_players)
    return means


df = ''


def TestModule():
    print('Hello World')


def Init(path):
    global df
    df = pd.read_excel(path)


def ShowRawData():
    global df
    print(df)


def nearestToOne(dfOriginal, one, infos):
    df = dfOriginal.copy()
    df['Distance'] = df.apply(lambda element: norme(element, one, infos), axis=1)
    df = df.sort_values(by='Distance', ascending=True)
    return df


def nearestToMany(dfOriginal, many, infos):
    for one in many:
        nearestToOne(dfOriginal, one, infos)


def normalize(val, max, min):
    return (val - min) / (max - min)


def findCommonData(list1, list2):
    return 1 if common_data(list1, list2) else 0


def getNearestKClass(train, player, infosList, col, K):
    topK = nearestToOne(train, player, infosList).head(K)
    val = ''
    for m in topK[col].mode():
        val += m + ','
    return val[:len(val)-1]


def evaluate_results(entity, predicted, wanted, dictionnary):
    result = 0
    for x in entity[predicted].split(','):
        for w in dictionnary[x]:
            if entity[wanted] in dictionnary[x][w]:
                result += w
    return result / len(entity[predicted].split(','))


def normalize(val: float, max: float, min: float) -> float:
    if (max == min):
        return 1.0
    else:
        return (val - min) / (max - min)


def cleanUpRowsWith(df: pd.DataFrame, dict: dict) -> pd.DataFrame:
    for el in dict['elements']:
        df = df[df[dict['col']] != el]
    return df


class Ml_1:
    train = None
    test = None
    normList = []
    oneHotList = []
    infosToCalculate = []

    def __init__(
        self,
        train=None,
        test=None,
        normList=[],
        oneHotList=[]
    ) -> None:
        self.train = train
        self.test = test
        self.normList = normList
        self.oneHotList = oneHotList
    
    def trainCleanup(self, dict_col_elements):
        if isinstance(self.train, pd.DataFrame):
            self.train = cleanUpRowsWith(self.train, dict_col_elements)

    def testCleanup(self, dict_col_elements):
        if isinstance(self.test, pd.DataFrame):
            self.test = cleanUpRowsWith(self.train, dict_col_elements)

    def setNormList(self, normList):
        self.normList = normList

    def setOneHotList(self, oneHotList):
        self.oneHotList = oneHotList

    def initTrainFromSQL(self, db, query):
        self.train = pd.read_sql_query(query, db)

    def initTrainFromCSV(self, path):
        self.train = pd.read_csv(path)

    def initTrainFromExcel(self, path):
        self.train = pd.read_excel(path)

    def initTrainFromJSON(self, path):
        self.train = pd.read_json(path)

    def initTrainFromDict(self, dict):
        self.train = pd.DataFrame.from_dict(dict)

    def initTestFromSQL(self, db, query):
        self.test = pd.read_sql_query(query, db)

    def initTestFromCSV(self, path):
        self.test = pd.read_csv(path)

    def initTestFromExcel(self, path):
        self.test = pd.read_excel(path)

    def initTestFromJSON(self, path):
        self.test = pd.read_json(path)

    def initTestFromDict(self, dict):
        self.test = pd.DataFrame.from_dict(dict)

    def normalizeTrain(self):
        for norm in self.normList:
            self.train[norm] = pd.to_numeric(self.train[norm])
            self.train[norm] = self.train[norm].fillna(0)
            self.train["Norm_" + str(norm)] = self.train.apply(
                lambda element: normalize(element[norm], self.train[norm].max(
                ), self.train[norm].min()),
                axis=1)
            self.infosToCalculate.append("Norm_" + str(norm))

    def oneHotTrain(self):
        for e in self.oneHotList:
            for val in self.train[e].unique():
                self.train['Norm_' + e + '_' + val] = self.train.apply(
                    lambda element: findCommonData(
                        [element[e]],
                        [val],
                    ),
                    axis=1)
                self.infosToCalculate = [
                    *self.infosToCalculate, *['Norm_' + e + '_' + val]
                ]

    def normalizeTest(self):
        for norm in self.normList:
            self.test[norm] = pd.to_numeric(self.test[norm])
            self.test[norm] = self.test[norm].fillna(0)
            self.test["Norm_" + str(norm)] = self.test.apply(
                lambda element: normalize(element[norm], self.train[norm].max(
                ), self.train[norm].min()),
                axis=1)

    def oneHotTest(self):
        for e in self.oneHotList:
            for val in self.train[e].unique():
                self.test['Norm_' + e + '_' + val] = self.test.apply(
                    lambda element: findCommonData(
                        [element[e]],
                        [val]
                    ),
                    axis=1)

    def KNN(self, col, K) -> str:
        self.test['predicted {}'.format(col)] = self.test.apply(
            lambda element: getNearestKClass(
                self.train,
                element,
                self.infosToCalculate,
                col,
                K
            ),
            axis=1
        )
        return self.test.loc[0, 'predicted {}'.format(col)]

    def gettrain(self) -> pd.DataFrame:
        return self.train

    def gettest(self) -> pd.DataFrame:
        return self.test

    def getInfosToCalculate(self) -> list:
        return self.infosToCalculate

    #debug
    def showTrain(self):
        print(self.train)

    def showTest(self):
        print(self.test)
        #for col in self.test.columns:
        #    print(str(col) + ' : ' + str(self.test.loc[0, col]))

    def showNormList(self):
        print(self.normList)

    def showOneHotList(self):
        print(self.oneHotList)

    def showInfosToCalculate(self):
        print(self.infosToCalculate)
