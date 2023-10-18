import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from mafese import get_dataset
from intelelm import MhaElmClassifier
from copy import deepcopy
import warnings
from scipy.stats import cauchy


warnings.filterwarnings("ignore")


PopSize = 20
DimSize = 10
LB = [-10] * DimSize
UB = [10] * DimSize

C1 = 0.2
C3 = 3
MU = 25
SIGMA = 3

TrialRuns = 30
MaxFEs = 1000

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

muF1, muF2, muF3 = 0.5, 0.5, 0.5
muCr1, muCr2, muCr3 = 0.5, 0.5, 0.5

X_train = None
y_train = None
Xtest = None
ytest = None
model = None

BestIndi = None
BestFit = None


def meanL(arr):
    numer = 0
    denom = 0
    for var in arr:
        numer += var ** 2
        denom += var
    return numer / denom


def fit_func(indi):
    global X_train, y_train, model
    return model.fitness_function(indi)


def score_func(indi):
    global Xtest, ytest, model
    return model.score(indi)


# initialize the Pop randomly
def Initialization():
    global Pop, FitPop, DimSize, BestIndi, BestFit
    Pop = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = -fit_func(Pop[i])
    best_idx = np.argmin(FitPop)
    BestIndi = deepcopy(Pop[np.argmin(FitPop)])
    BestFit = FitPop[best_idx]


def HADE():
    global Pop, FitPop, curIter, MaxIter, LB, UB, PopSize, DimSize, BestIndi, FitBest, muF1, muF2, muF3, muCr1, muCr2, muCr3
    c = 0.1
    F1_list, F2_list, F3_list = [], [], []
    Cr1_list, Cr2_list, Cr3_list = [], [], []

    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    sort_idx = np.argsort(FitPop)
    # 0-10: superior; 10-50: borderline; 50-90: borderline; 90-100: inferior
    for i in range(PopSize):
        idx = sort_idx[i]
        if i < 10:
            Off[idx] = Off[idx] + np.random.uniform(-1, 1, DimSize)  # Local search
            Off[idx] = np.clip(Off[idx], LB, UB)
            FitOff[idx] = fit_func(Off[idx])

        elif i < 50:
            f1 = cauchy.rvs(muF1, 0.1)  # F adaptation
            while True:
                if f1 > 1:
                    f1 = 1
                    break
                elif f1 < 0:
                    f1 = cauchy.rvs(muF1, 0.1)
                break

            cr1 = np.random.normal(muCr1, 0.1)  # Cr adaptation
            cr1 = np.clip(cr1, 0, 1)

            candi = list(range(0, PopSize))
            candi.remove(idx)
            r1, r2 = np.random.choice(candi, 2, replace=False)
            pbest_idx = sort_idx[0:np.random.randint(1, 20)]
            MEAN = np.mean(Pop[pbest_idx], axis=0)
            Off[idx] = Pop[idx] + f1 * (MEAN - Pop[idx]) + f1 * (Pop[r1] - Pop[r2])

            jrand = np.random.randint(0, DimSize)
            for j in range(DimSize):  # DE binomial crossover
                if np.random.rand() < cr1 or j == jrand:
                    pass
                else:
                    Off[idx][j] = Pop[idx][j]
            Off[idx] = np.clip(Off[idx], LB, UB)

            FitOff[idx] = fit_func(Off[idx])
            if FitOff[idx] < FitPop[idx]:  # Hyperparameter adaptation
                F1_list.append(f1)
                Cr1_list.append(cr1)

        elif i < 90:
            f2 = cauchy.rvs(muF2, 0.1)  # F adaptation
            while True:
                if f2 > 1:
                    f2 = 1
                    break
                elif f2 < 0:
                    f2 = cauchy.rvs(muF2, 0.1)
                break

            cr2 = np.random.normal(muCr2, 0.1)  # Cr adaptation
            cr2 = np.clip(cr2, 0, 1)

            candi = list(range(0, PopSize))
            candi.remove(idx)
            r1, r2 = np.random.choice(candi, 2, replace=False)

            pbest_idx = sort_idx[0:np.random.randint(1, 20)]
            MEAN = np.mean(Pop[pbest_idx], axis=0)
            Off[idx] = Pop[idx] + f2 * (MEAN - Pop[idx]) + f2 * (Pop[r1] - Pop[r2])

            jrand = np.random.randint(0, DimSize)
            for j in range(DimSize):  # DE binomial crossover
                if np.random.rand() < cr2 or j == jrand:
                    pass
                else:
                    Off[idx][j] = Pop[idx][j]
            Off[idx] = np.clip(Off[idx], LB, UB)

            FitOff[idx] = fit_func(Off[idx])
            if FitOff[idx] < FitPop[idx]:  # Hyperparameter adaptation
                F2_list.append(f2)
                Cr2_list.append(cr2)
        else:
            f3 = cauchy.rvs(muF3, 0.1)  # F adaptation
            while True:
                if f3 > 1:
                    f3 = 1
                    break
                elif f3 < 0:
                    f3 = cauchy.rvs(muF3, 0.1)
                break

            cr3 = np.random.normal(muCr3, 0.1)  # Cr adaptation
            cr3 = np.clip(cr3, 0, 1)

            candi = list(range(0, PopSize))
            candi.remove(idx)
            r1, r2 = np.random.choice(candi, 2, replace=False)
            pbest_idx = sort_idx[0:np.random.randint(1, 20)]
            MEAN = np.mean(Pop[pbest_idx], axis=0)
            Off[idx] = Pop[idx] + f3 * (MEAN - Pop[idx]) + f3 * (Pop[r1] - Pop[r2])

            jrand = np.random.randint(0, DimSize)
            for j in range(DimSize):  # DE binomial crossover
                if np.random.rand() < cr3 or j == jrand:
                    pass
                else:
                    Off[idx][j] = Pop[idx][j]
            Off[idx] = np.clip(Off[idx], LB, UB)

            FitOff[idx] = fit_func(Off[idx])
            if FitOff[idx] < FitPop[idx]:  # Hyperparameter adaptation
                F3_list.append(f3)
                Cr3_list.append(cr3)

        if FitOff[idx] < FitPop[idx]:
            FitPop[idx] = FitOff[idx]
            Pop[idx] = deepcopy(Off[idx])
            if FitOff[idx] < FitBest:
                FitBest = FitOff[idx]
                BestIndi = deepcopy(Off[idx])

    if len(F1_list) == 0:
        pass
    else:
        muF1 = (1 - c) * muF1 + c * meanL(F1_list)
    if len(F2_list) == 0:
        pass
    else:
        muF2 = (1 - c) * muF2 + c * meanL(F2_list)
    if len(F3_list) == 0:
        pass
    else:
        muF3 = (1 - c) * muF3 + c * meanL(F3_list)

    if len(Cr1_list) == 0:
        pass
    else:
        muCr1 = (1 - c) * muCr1 + c * np.mean(Cr1_list)
    if len(Cr2_list) == 0:
        pass
    else:
        muCr2 = (1 - c) * muCr2 + c * np.mean(Cr2_list)
    if len(Cr3_list) == 0:
        pass
    else:
        muCr3 = (1 - c) * muCr3 + c * np.mean(Cr3_list)


def RunHADE(setname):
    global curIter, TrialRuns, Pop, FitPop, DimSize, X_train, y_train, Xtest, ytest, model, DimSize, LB, UB, BestIndi, BestIndi, muF1, muF2, muF3, muCr1, muCr2, muCr3
    muF1, muF2, muF3 = 0.5, 0.5, 0.5
    muCr1, muCr2, muCr3 = 0.5, 0.5, 0.5
    dataset = get_dataset(setname)
    X = dataset.X
    y = dataset.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=100)

    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    le_y = LabelEncoder()
    le_y.fit(y)
    y_train = le_y.transform(y_train)
    y_test = le_y.transform(y_test)

    All_Trial_Best = []
    All_score = []
    for i in range(TrialRuns):
        BestIndi, BestIndi = None, None

        model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="AS")
        model.network, model.obj_scaler = model.create_network(X_train, y_train)
        y_scaled = model.obj_scaler.transform(y_train)
        model.X_temp, model.y_temp = X_train, y_scaled

        DimSize = len(X_train[0]) * 10 + 10
        Pop = np.zeros((PopSize, DimSize))
        LB = [-10] * DimSize
        UB = [10] * DimSize

        Best_list = []
        curIter = 0
        Initialization()
        Best_list.append(min(FitPop))
        np.random.seed(2022 + 88 * i)
        while curIter <= MaxIter:
            HADE()
            curIter += 1
            Best_list.append(min(FitPop))
            # print("Iter: ", curIter, "Best: ", Fgbest)
        model.network.update_weights_from_solution(BestIndi, model.X_temp, model.y_temp)
        All_score.append(model.score(X_test, y_test))
        All_Trial_Best.append(np.abs(Best_list))
    np.savetxt("./HADE_Data/Loss/" + str(FuncNum) + ".csv", All_Trial_Best, delimiter=",")
    np.savetxt("./HADE_Data/AS/" + str(FuncNum) + ".csv", All_score, delimiter=",")


def main(setname):
    global FuncNum
    FuncNum = setname
    RunHADE(setname)


if __name__ == "__main__":
    if os.path.exists('./HADE_Data/AS') == False:
        os.makedirs('./HADE_Data/AS')
    if os.path.exists('./HADE_Data/Loss') == False:
        os.makedirs('./HADE_Data/Loss')
    Datasets = ['aggregation', 'aniso', 'appendicitis', 'balance', 'banknote', 'blobs', 'Blood',
                'BreastCancer', 'BreastEW', 'circles', 'CongressEW', 'diagnosis_II', 'Digits', 'ecoli', 'flame',
                'Glass', 'heart', 'HeartEW', 'Hill-valley', 'Horse', 'Ionosphere', 'Iris', 'jain', 'liver',
                'Madelon', 'Monk1', 'Monk2', 'Monk3', 'moons', 'mouse', 'pathbased', 'seeds', 'smiley',
                'Sonar', 'Soybean-small', 'SpectEW', 'Tic-tac-toe', 'varied', 'vary-density', 'vertebral2', 'Vowel',
                'WaveformEW', 'wdbc', 'Wine', 'Zoo']
    for setname in Datasets:
        main(setname)
