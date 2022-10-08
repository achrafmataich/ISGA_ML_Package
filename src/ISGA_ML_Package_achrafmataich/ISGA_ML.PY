import numpy as np
import pandas as pd
from math import sqrt, pow


def common_data(list1, list2):
    result = False

    # traverse in the 1st list
    for x in list1:

        # traverse in the 2nd list
        for y in list2:

            # if one common
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
    df['Distance'] = df.apply(lambda player: norme(player, one, infos), axis=1)
    df = df.sort_values(by='Distance', ascending=True)
    return df


def nearestToMany(dfOriginal, many, infos):
    for one in many:
        nearestToOne(dfOriginal, one, infos)


def RankPlayers1():
    global df
    infos_to_calculate = ['Norm_Age', 'Norm_Height', 'Norm_Weight',
                          'Norm_PositionMF', 'Norm_PositionFW', 'Norm_PositionDF', 'Norm_PositionGK']
    df_cpy = df.copy()
    for index, player in df_cpy.iterrows():
        print(index, '. ', player.Name)
    i_players = input('choisissez l\'indice du joueur a choisir : ').split(',')
    print(i_players)

    df_cpy['Norm_Age'] = df_cpy.apply(lambda player: (
        player['Age'] - df_cpy['Age'].min()) / (df_cpy['Age'].max() - df_cpy['Age'].min()), axis=1)
    df_cpy['Norm_Height'] = df_cpy.apply(lambda player: (
        player['Height'] - df_cpy['Height'].min()) / (df_cpy['Height'].max() - df_cpy['Height'].min()), axis=1)
    df_cpy['Norm_Weight'] = df_cpy.apply(lambda player: (
        player['Weight'] - df_cpy['Weight'].min()) / (df_cpy['Weight'].max() - df_cpy['Weight'].min()), axis=1)
    df_cpy['Norm_PositionMF'] = df_cpy.apply(lambda player: 1 if common_data(
        player['Position'].split(','), ['MF']) else 0, axis=1)
    df_cpy['Norm_PositionFW'] = df_cpy.apply(lambda player: 1 if common_data(
        player['Position'].split(','), ['FW']) else 0, axis=1)
    df_cpy['Norm_PositionDF'] = df_cpy.apply(lambda player: 1 if common_data(
        player['Position'].split(','), ['DF']) else 0, axis=1)
    df_cpy['Norm_PositionGK'] = df_cpy.apply(lambda player: 1 if common_data(
        player['Position'].split(','), ['GK']) else 0, axis=1)
    selectedPlayers = []
    for el in i_players:
        selectedPlayers.append(df_cpy.iloc[[el]])
    modeldict = getCenterDot(selectedPlayers, infos_to_calculate)
    df_cpy['Distance'] = df_cpy.apply(lambda player: norme(
        player, modeldict, infos_to_calculate), axis=1)
    df_cpy = df_cpy.sort_values(by='Distance', ascending=True)
    print(df_cpy)


def RankPlayers2():
    global df
    infos_to_calculate = ['Norm_Age', 'Norm_Height', 'Norm_Weight',
                          'Norm_PositionMF', 'Norm_PositionFW', 'Norm_PositionDF', 'Norm_PositionGK']
    df_cpy = df.copy()
    for index, player in df_cpy.iterrows():
        print(index, '. ', player.Name)
    i_players = input('choisissez l\'indice du joueur a choisir : ').split(',')
    print(i_players)

    df_cpy['Norm_Age'] = df_cpy.apply(lambda player: (
        player['Age'] - df_cpy['Age'].min()) / (df_cpy['Age'].max() - df_cpy['Age'].min()), axis=1)
    df_cpy['Norm_Height'] = df_cpy.apply(lambda player: (
        player['Height'] - df_cpy['Height'].min()) / (df_cpy['Height'].max() - df_cpy['Height'].min()), axis=1)
    df_cpy['Norm_Weight'] = df_cpy.apply(lambda player: (
        player['Weight'] - df_cpy['Weight'].min()) / (df_cpy['Weight'].max() - df_cpy['Weight'].min()), axis=1)
    df_cpy['Norm_PositionMF'] = df_cpy.apply(lambda player: 1 if common_data(
        player['Position'].split(','), ['MF']) else 0, axis=1)
    df_cpy['Norm_PositionFW'] = df_cpy.apply(lambda player: 1 if common_data(
        player['Position'].split(','), ['FW']) else 0, axis=1)
    df_cpy['Norm_PositionDF'] = df_cpy.apply(lambda player: 1 if common_data(
        player['Position'].split(','), ['DF']) else 0, axis=1)
    df_cpy['Norm_PositionGK'] = df_cpy.apply(lambda player: 1 if common_data(
        player['Position'].split(','), ['GK']) else 0, axis=1)
    selectedPlayers = []
    for el in i_players:
        selectedPlayers.append(df_cpy.iloc[[el]])

    nearestToMany(df_cpy, selectedPlayers, infos_to_calculate)

# KNN Algorithm


def rank_KNN(nanList, dropList, normList, oneHotList, k):
    global df

    df_cpy = df.copy()
    df_cpy = df_cpy.head(100)
    # fill nan values with zeros
    for e in nanList:
        df_cpy[e] = df_cpy[e].fillna(0)
    # drop columns that are useless
    df_cpy = df_cpy.drop(dropList, axis=1)
    print(df_cpy)
    infosList = []
    # normalization
    for e in normList:
        df_cpy['Norm_'+e] = df_cpy.apply(lambda player: (
            player[e] - df_cpy[e].min()) / (df_cpy[e].max() - df_cpy[e].min()), axis=1)
        infosList = [*infosList, *['Norm_'+e]]

    # one hot encoding
    for e in oneHotList:
        for val in df_cpy[e].unique():
            df_cpy['Norm_'+e+'_'+val] = df_cpy.apply(
                lambda player: 1 if common_data([player[e]], [val]) else 0, axis=1)
            infosList = [*infosList, *['Norm_'+e+'_'+val]]

    df_cpy = df_cpy.sample(frac=1).reset_index(drop=True)
    print(df_cpy)
    train, reste, test = np.split(
        df_cpy, [int(0.9*len(df_cpy)), int(0.9*len(df_cpy))])
    print(train)
    print('--------------------------------------')
    print(test)

    for i in range(0, len(test)):
        topK = nearestToOne(train, test.iloc[[i]], infosList).head(k)
        print(topK['Label'].mode()[0], ' for ', test.iloc[[i]]['short_name'])

    df_cpy.to_excel('data/output.xlsx', index=False)


def normalize(val, max, min):
    return (val - min) / (max - min)


def findCommonData(list1, list2):
    return 1 if common_data(list1, list2) else 0


def getNearestK(train, player, infosList, col):
    topK = nearestToOne(train, player, infosList).head(
        (len(train[col].unique())*2)+1)
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


def rank_KNN_position(nanList, dropList, normList, oneHotList, k):
    global df
    #[ 'RM' 'CAM' 'RCB' 'RCM' 'RF' 'LCB' 'RB' 'CDM' 'LM' 'LB' 'CB' 'LWB' 'CM' 'RAM' 'RWB' 'LAM']
    positions_weights = {
        # Goal Keeper
        'GK': {
            1: ['GK'],
            0: ['ST', 'RCB', 'LM', 'RS', 'LS', 'LB', 'LW', 'LCM', 'LCB', 'LF', 'RM', 'CDM', 'LDM', 'CB', 'RB', 'CAM', 'RCM', 'RWB', 'RW', 'RF', 'RDM', 'CF', 'LWB', 'RAM', 'CM', 'LAM']
        },
        # FORWARD
        #   Strikers
        'ST': {
            1: ['ST'],
            0.75: ['RS', 'LS'],
            0.5: ['CF', 'LF', 'RF', 'LW', 'RW'],
            0.25: ['LAM', 'RAM', 'CAM'],
            0: ['GK', 'RCB', 'LM', 'LB', 'LCM', 'LCB', 'RM', 'CDM', 'LDM', 'CB', 'RB', 'RCM', 'RWB', 'RDM', 'LWB', 'CM']
        },
        'LS': {
            1: ['LS'],
            0.75: ['LW', 'LF', 'ST'],
            0.5: ['CF', 'RS'],
            0.25: ['RW', 'LAM'],
            0: ['GK', 'RCB', 'LM', 'LB', 'LCM', 'LCB', 'RM', 'CDM', 'LDM', 'CB', 'RB', 'CAM', 'RCM', 'RWB', 'RF', 'RDM', 'LWB', 'CM', 'RAM']
        },
        'RS': {
            1: ['RS'],
            0.75: ['RW', 'RF', 'ST'],
            0.5: ['CF', 'LS'],
            0.25: ['LW', 'RAM'],
            0: ['GK', 'RCB', 'LM', 'LB', 'LCM', 'LCB', 'LF', 'RM', 'CDM', 'LDM', 'CB', 'RB', 'CAM', 'RCM', 'RWB', 'RDM', 'LWB', 'CM', 'LAM']
        },
        #   Forwards
        'CF': {
            1: ['CF'],
            0.75: ['LF', 'RF'],
            0.5: ['ST', 'LS', 'RS'],
            0.25: ['LW', 'RW', 'CAM'],
            0: ['GK', 'RCB', 'LM', 'LB', 'LCM', 'LCB', 'RM', 'CDM', 'LDM', 'CB', 'RB', 'RCM', 'RWB', 'RDM', 'LWB', 'RAM', 'CM', 'LAM']
        },
        'LF': {
            1: ['LF'],
            0.75: ['CF', 'LS', 'LW', 'RF'],
            0.5: ['ST'],
            0.25: ['LM', 'RW', 'LAM'],
            0: ['GK', 'RCB', 'RS', 'LB', 'LCM', 'LCB', 'RM', 'CDM', 'LDM', 'CB', 'RB', 'CAM', 'RCM', 'RWB', 'RDM', 'LWB', 'RAM', 'CM']
        },
        'RF': {
            1: ['RF'],
            0.75: ['CF', 'RS', 'RW', 'LF'],
            0.5: ['ST'],
            0.25: ['RM', 'LW', 'RAM'],
            0: ['GK', 'RCB', 'LS', 'LB', 'LCM', 'LCB', 'LM', 'CDM', 'LDM', 'CB', 'RB', 'CAM', 'RCM', 'RWB', 'RDM', 'LWB', 'LAM', 'CM']
        },
        #   Wings
        'LW': {
            1: ['LW'],
            0.75: ['LF', 'LAM', 'LS'],
            0.5: ['LM', 'ST'],
            0.25: ['CF', 'RW', 'RF', 'RS', 'RAM', 'LWB'],
            0: ['GK', 'LDM', 'LCM', 'RM', 'CAM', 'RCB', 'RDM', 'RCM', 'LCB', 'RB', 'CDM', 'LB', 'CB', 'CM', 'RWB']
        },
        'RW': {
            1: ['RW'],
            0.75: ['RF', 'RAM', 'RS'],
            0.5: ['RM', 'ST'],
            0.25: ['CF', 'LW', 'LF', 'LS', 'LAM', 'RWB'],
            0: ['GK', 'LDM', 'LCM', 'LM', 'CAM', 'RCB', 'RDM', 'RCM', 'LCB', 'RB', 'CDM', 'LB', 'CB', 'CM', 'LWB']
        },
        # MIDDLEFIELD
        #   Attacking Middlefielders
        'CAM': {
            1: ['CAM'],
            0.75: ['LAM', 'RAM'],
            0.5: ['LM', 'RM'],
            0.25: ['CF', 'ST', 'LCM', 'RCM', 'CM'],
            0: ['GK', 'ST', 'RCB', 'LM', 'RS', 'LS', 'LB', 'LW', 'LCM', 'LCB', 'LF', 'RM', 'CDM', 'LDM', 'CB', 'RB', 'CAM', 'RCM', 'RWB', 'RW', 'RF', 'RDM', 'CF', 'LWB', 'RAM', 'CM', 'LAM']
        },
        'LAM': {
            1: ['LAM'],
            0.75: ['CAM', 'LW', 'LM', 'RAM'],
            0.5: ['LCM'],
            0.25: ['ST', 'LS', 'LF', 'RW', 'CM'],
            0: ['GK', 'RCB', 'RS', 'LB', 'LCB', 'RM', 'CDM', 'LDM', 'CB', 'RB', 'RCM', 'RWB', 'RF', 'RDM', 'CF', 'LWB']
        },
        'RAM': {
            1: ['RAM'],
            0.75: ['CAM', 'RW', 'RM', 'LAM'],
            0.5: ['RCM'],
            0.25: ['ST', 'RS', 'RF', 'LW', 'CM'],
            0: ['GK', 'RCB', 'LS', 'LB', 'LCB', 'LM', 'CDM', 'LDM', 'CB', 'RB', 'LCM', 'RWB', 'LF', 'RDM', 'CF', 'LWB']
        },
        #   Middlefielders
        'CM': {
            1: ['CM'],
            0.75: ['LCM', 'RCM'],
            0.5: ['LM', 'RM'],
            0.25: ['LDM', 'RDM', 'CDM', 'LAM', 'RAM', 'CAM'],
            0: ['GK', 'ST', 'RCB', 'RS', 'LS', 'LB', 'LW', 'LCB', 'LF', 'CB', 'RB', 'RWB', 'RW', 'RF', 'CF', 'LWB']
        },
        'LCM': {
            1: ['LCM'],
            0.75: ['CM', 'LM', 'RCM'],
            0.5: ['LAM'],
            0.25: ['LDM', 'CAM'],
            0: ['GK', 'ST', 'RCB', 'RS', 'LS', 'LB', 'LW', 'LCB', 'LF', 'RM', 'CDM', 'CB', 'RB', 'RWB', 'RW', 'RF', 'RDM', 'CF', 'LWB', 'RAM']
        },
        'RCM': {
            1: ['RCM'],
            0.75: ['CM', 'RM', 'LCM'],
            0.5: ['RAM'],
            0.25: ['RDM', 'CAM'],
            0: ['GK', 'ST', 'RCB', 'RS', 'LS', 'LB', 'LW', 'LCB', 'LF', 'LM', 'CDM', 'CB', 'RB', 'RWB', 'RW', 'RF', 'LDM', 'CF', 'LWB', 'LAM']
        },
        'RM': {
            1: ['RM'],
            0.75: ['RCM', 'RAM'],
            0.5: ['RW', 'CM', 'LM', 'CAM'],
            0.25: ['RF', 'RWB'],
            0: ['GK', 'ST', 'RCB', 'RS', 'LS', 'LB', 'LW', 'LCM', 'LCB', 'LF', 'CDM', 'LDM', 'CB', 'RB', 'RDM', 'CF', 'LWB', 'LAM']
        },
        'LM': {
            1: ['LM'],
            0.75: ['LCM', 'LAM'],
            0.5: ['LW', 'CM', 'RM', 'CAM'],
            0.25: ['LF', 'LWB'],
            0: ['GK', 'ST', 'RCB', 'RS', 'LS', 'LB', 'RW', 'RCM', 'LCB', 'RF', 'CDM', 'LDM', 'CB', 'RB', 'RWB', 'RDM', 'CF', 'RAM']
        },
        #   Deffensive Middlefielders
        'CDM': {
            1: ['CDM'],
            0.75: ['LDM', 'RDM'],
            0.25: ['CM', 'LWB', 'RWB'],
            0: ['GK', 'ST', 'RCB', 'LM', 'RS', 'LS', 'LB', 'LW', 'LCM', 'LCB', 'LF', 'RM', 'CB', 'RB', 'CAM', 'RCM', 'RW', 'RF', 'CF', 'RAM']
        },
        'LDM': {
            1: ['LDM'],
            0.75: ['CDM'],
            0.5: ['RDM', 'LWB'],
            0.25: ['LB', 'LCM', 'CM'],
            0: ['GK', 'ST', 'LW', 'LM', 'RM', 'CAM', 'RW', 'RCB', 'RS', 'LS', 'RCM', 'RF', 'LCB', 'RB', 'CB', 'LF', 'RAM', 'RWB', 'LAM']
        },
        'RDM': {
            1: ['RDM'],
            0.75: ['CDM'],
            0.5: ['LDM', 'RWB'],
            0.25: ['RB', 'RCM', 'CM'],
            0: ['GK', 'ST', 'LW', 'LCM', 'RM', 'CAM', 'RW', 'RCB', 'RS', 'LS', 'RM', 'RF', 'LCB', 'LB', 'CB', 'LF', 'RAM', 'LWB', 'LAM']
        },
        # BACK
        #   Backs
        'CB': {
            1: ['CB'],
            0.75: ['LCB', 'RCB'],
            0.25: ['LB', 'RB'],
            0: ['GK', 'ST', 'LM', 'RS', 'LS', 'LW', 'LCM', 'LF', 'RM', 'CDM', 'LDM', 'CAM', 'RCM', 'RWB', 'RW', 'RF', 'RDM', 'CF', 'LWB', 'RAM', 'CM', 'LAM']
        },
        'LCB': {
            1: ['LCB'],
            0.75: ['CB', 'RCB'],
            0.5: ['LB'],
            0.25: ['RB'],
            0: ['GK', 'ST', 'LM', 'RS', 'LS', 'LW', 'LCM', 'LF', 'RM', 'CDM', 'LDM', 'CAM', 'RCM', 'RWB', 'RW', 'RF', 'RDM', 'CF', 'LWB', 'RAM', 'CM', 'LAM']
        },
        'RCB': {
            1: ['RCB'],
            0.75: ['CB', 'LCB'],
            0.5: ['RB'],
            0.25: ['LB'],
            0: ['GK', 'ST', 'LM', 'RS', 'LS', 'LW', 'LCM', 'LF', 'RM', 'CDM', 'LDM', 'CAM', 'RCM', 'RWB', 'RW', 'RF', 'RDM', 'CF', 'LWB', 'RAM', 'CM', 'LAM']
        },
        'LB': {
            1: ['LB'],
            0.75: ['LWB'],
            0.5: ['LCB'],
            0.25: ['CB', 'RCB', 'RB'],
            0: ['GK', 'ST', 'LM', 'RS', 'LS', 'LW', 'LCM', 'LF', 'RM', 'CDM', 'LDM', 'CAM', 'RCM', 'RWB', 'RW', 'RF', 'RDM', 'CF', 'RAM', 'CM', 'LAM']
        },
        'RB': {
            1: ['RB'],
            0.75: ['RWB'],
            0.5: ['RCB'],
            0.25: ['CB', 'LCB', 'LB'],
            0: ['GK', 'ST', 'LM', 'RS', 'LS', 'LW', 'LCM', 'LF', 'RM', 'CDM', 'LDM', 'CAM', 'RCM', 'LWB', 'RW', 'RF', 'RDM', 'CF', 'RAM', 'CM', 'LAM']
        },
        'LWB': {
            1: ['LWB'],
            0.75: ['LB'],
            0.5: ['LDM'],
            0.25: ['CDM', 'LM'],
            0: ['GK', 'ST', 'RCB', 'LM', 'RS', 'LS', 'LW', 'LCM', 'LCB', 'LF', 'RM', 'CB', 'RB', 'CAM', 'RCM', 'RWB', 'RW', 'RF', 'RDM', 'CF', 'RAM', 'CM', 'LAM']
        },
        'RWB': {
            1: ['RWB'],
            0.75: ['RB'],
            0.5: ['RDM'],
            0.25: ['CDM', 'RM'],
            0: ['GK', 'ST', 'RCB', 'LM', 'RS', 'LS', 'LW', 'LCM', 'LCB', 'LF', 'RM', 'CB', 'LB', 'CAM', 'RCM', 'LWB', 'RW', 'RF', 'LDM', 'CF', 'RAM', 'CM', 'LAM']
        }
    }

    print(positions_weights['RDM'])

    df_cpy = df.copy().head(500)
    # fill nan values with zeros
    for e in nanList:
        df_cpy[e] = df_cpy[e].fillna(0)
    # drop columns that are useless
    df_cpy = df_cpy.drop(dropList, axis=1)
    infosList = []
    # drop rows that can make us problems
    df_cpy = df_cpy[df_cpy['club_position'].notnull()]
    df_cpy = df_cpy[df_cpy['club_position'] != 'SUB']
    df_cpy = df_cpy[df_cpy['club_position'] != 'RES']
    # Shuffle data
    df_cpy = df_cpy.sample(frac=1).reset_index(drop=True)
    # normalization
    print('\n---------------------------\nNormalization')
    for e in normList:
        print('Normalisation de la variable "' + e + '"..')
        df_cpy['Norm_'+e] = df_cpy.apply(
            lambda player: normalize(
                player[e],
                df_cpy[e].max(),
                df_cpy[e].min()
            ),
            axis=1
        )
        df_cpy = df_cpy.drop([e], axis=1)
        infosList = [*infosList, *['Norm_'+e]]

    # one hot encoding
    print('\n---------------------------\nOne Hot Encoding')
    for e in oneHotList:
        for val in df_cpy[e].unique():
            df_cpy['Norm_'+e+'_'+val] = df_cpy.apply(
                lambda player: findCommonData(
                    [player[e]],
                    [val]
                ),
                axis=1
            )
            infosList = [*infosList, *['Norm_'+e+'_'+val]]
        df_cpy = df_cpy.drop([e], axis=1)

    # split data to train and test
    #train = df_cpy[df_cpy['club_position'] != 'SUB']
    #test = df_cpy[df_cpy['club_position'] == 'SUB']
    train, reste, test = np.split(
        df_cpy, [int(0.75*len(df_cpy)), int(0.75*len(df_cpy))])

    print(train)
    print(test)
    test['predicted_pos'] = test.apply(
        lambda player: getNearestK(train, player, infosList),
        axis=1
    )
    test.drop(infosList, axis=1).to_excel('/python/data/out.xlsx', index=False)
    result = pd.read_excel('/python/data/out.xlsx')
    result['accuracy'] = result.apply(
        lambda player : evaluate_results(
            player,
            'predicted_pos',
            'club_position',
            positions_weights,
        ),
        axis=1
    )
    result.to_excel('/python/data/finalOut.xlsx', index=False)
    print('END')

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
            print("NORMALIZING " + str(norm))
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

    def KNN(self, col) -> str:
        self.test['predicted {}'.format(col)] = self.test.apply(
            lambda element: getNearestK(self.train, element, self.
                                        infosToCalculate, col),
            axis=1)
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
