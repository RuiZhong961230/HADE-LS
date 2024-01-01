import os
from copy import deepcopy
import numpy as np
from scipy.stats import cauchy
from opfunu.cec_based import cec2013


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = DimSize * 1000

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

Func_num = 0

BestIndi = None
FitBest = float("inf")

muF1, muF2, muF3 = 0.5, 0.5, 0.5
muCr1, muCr2, muCr3 = 0.5, 0.5, 0.5


def meanL(arr):
    numer = 0
    denom = 0
    for var in arr:
        numer += var ** 2
        denom += var
    return numer / denom


def Initial(func):
    global Pop, FitPop, DimSize, BestIndi, FitBest
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func.evaluate(Pop[i])
    FitBest = min(FitPop)
    BestIndi = deepcopy(Pop[np.argmin(FitPop)])


def HADE(func):
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
            Off[idx] = Pop[idx] + np.random.uniform(-1, 1, DimSize)  # Local search
            Off[idx] = np.clip(Off[idx], LB, UB)
            FitOff[idx] = func.evaluate(Off[idx])

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

            FitOff[idx] = func.evaluate(Off[idx])
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

            FitOff[idx] = func.evaluate(Off[idx])
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

            FitOff[idx] = func.evaluate(Off[idx])
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


def RunHADE(func):
    global curIter, MaxIter, TrialRuns, Pop, FitPop, DimSize, muF1, muF2, muF3, muCr1, muCr2, muCr3
    All_Trial_Best = []
    for i in range(TrialRuns):
        muF1, muF2, muF3 = 0.5, 0.5, 0.5
        muCr1, muCr2, muCr3 = 0.5, 0.5, 0.5
        Best_list = []
        curIter = 0
        Initial(func)
        Best_list.append(FitBest)
        np.random.seed(2022 + 88 * i)
        while curIter < MaxIter:
            HADE(func)
            curIter += 1
            Best_list.append(FitBest)
        All_Trial_Best.append(Best_list)
    np.savetxt("./HADE_Data/CEC2013/F" + str(Func_num) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global Func_num, DimSize, Pop, MaxFEs, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    LB = [-100] * dim
    UB = [100] * dim

    CEC2013 = [cec2013.F12013(Dim), cec2013.F22013(Dim), cec2013.F32013(Dim), cec2013.F42013(Dim), cec2013.F52013(Dim),
               cec2013.F62013(Dim), cec2013.F72013(Dim), cec2013.F82013(Dim), cec2013.F92013(Dim), cec2013.F102013(Dim),
               cec2013.F112013(Dim), cec2013.F122013(Dim), cec2013.F132013(Dim), cec2013.F142013(Dim),
               cec2013.F152013(Dim), cec2013.F162013(Dim), cec2013.F172013(Dim), cec2013.F182013(Dim),
               cec2013.F192013(Dim), cec2013.F202013(Dim), cec2013.F212013(Dim), cec2013.F222013(Dim),
               cec2013.F232013(Dim), cec2013.F242013(Dim), cec2013.F252013(Dim), cec2013.F262013(Dim),
               cec2013.F272013(Dim), cec2013.F282013(Dim)]

    Func_num = 0
    for i in range(len(CEC2013)):
        Func_num = i + 1
        RunHADE(CEC2013[i])


if __name__ == "__main__":
    if os.path.exists('HADE_Data/CEC2013') == False:
        os.makedirs('HADE_Data/CEC2013')
    Dims = [10, 30]
    for Dim in Dims:
        main(Dim)
