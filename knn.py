import scipy.io
from scipy.fft import fft
import pandas as pd
import numpy as np
import sys
import scipy.signal
from skfeature.function.similarity_based import fisher_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from progress.bar import Bar

np.set_printoptions(threshold=sys.maxsize)

fs = 500
patient = 1
HC = 0

def bandpower(x, fmin, fmax):

    Pxx = fft(x)
    
    ind_min = fmin*3 
    ind_max = fmax*3

    Pxx_abs = np.abs(Pxx)
    Pxx_pow = np.square(Pxx_abs)

    Pxx_sum = sum(Pxx_pow[ind_min: ind_max])

    return Pxx_sum

def segmentation(x, time_length, time_shift):

    length = time_length * 500
    shift_num = time_shift * 500

    epoches_array = []

    index = 0

    while True:

        if index*shift_num+length >= 45000:
            break
        else:

            array = x[index*shift_num:index*shift_num+length]
            epoches_array.append(array)

            index += 1

    #print(len(epoches_array))

    return epoches_array

def compute(x):

    band_array = []
    band_array_unused = []
    rp_ratio_array = []

    delta_com, theta_com, alpha_com, beta_com, gamma_com = [], [], [], [], []

    for channel in range(0, 30):

        channelData = np.asarray(x[channel])

        epoches_array = segmentation(channelData, 3, 1)

        delta_array, theta_array, alpha_array, beta_array, gamma_array, total_power_array = [], [], [], [], [], []

        for epoch in epoches_array:

            delta = bandpower(epoch, 1, 4)
            theta = bandpower(epoch, 4, 8)
            alpha = bandpower(epoch, 8, 13)
            beta = bandpower(epoch, 13, 30)
            gamma = bandpower(epoch, 30, 45)

            total_power = bandpower(epoch, 1, 45)

            delta_array.append(delta)
            theta_array.append(theta)
            alpha_array.append(alpha)
            beta_array.append(beta)
            gamma_array.append(gamma)

            total_power_array.append(total_power)

        delta_avg = np.mean(delta_array)
        theta_avg = np.mean(theta_array)
        alpha_avg = np.mean(alpha_array)
        beta_avg = np.mean(beta_array)
        gamma_avg = np.mean(gamma_array)

        total_power_avg = np.mean(total_power_array)

        rp_ratio_array.append([delta_avg/total_power_avg, theta_avg/total_power_avg, alpha_avg/total_power_avg, beta_avg/total_power_avg, gamma_avg/total_power_avg])
        band_array_unused.append([delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg])

        delta_com.append(delta_avg)
        theta_com.append(theta_avg)
        alpha_com.append(alpha_avg)
        beta_com.append(beta_avg)
        gamma_com.append(gamma_avg)

    band_array.extend(delta_com)
    band_array.extend(theta_com)
    band_array.extend(alpha_com)
    band_array.extend(beta_com)
    band_array.extend(gamma_com)

    band_array = np.asarray(band_array, dtype=float)
    rp_ratio_array = np.asarray(rp_ratio_array)

    #band_DF = pd.DataFrame(band_array)
    #print(band_DF)

    #ratio_DF = pd.DataFrame(rp_ratio_array)

    rp_features = band_array
    #rp_features = relative_power_lab(rp_ratio_array)

    return rp_features

def fisher(data_raw):

    train_features = []
    train_labels = []

    for index, patient_subjects in enumerate(data_raw[0:23]):
    
        features = compute(patient_subjects[0])

        if index < 13:
            train_features.append(features)
            train_labels.append(patient)

    for index, HC_subjects in enumerate(data_raw[23:47]):

        features = compute(HC_subjects[0])

        if index < 14:
            train_features.append(features)
            train_labels.append(HC)

    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels)

    band_DF = pd.DataFrame(train_features)
    #print(band_DF)

    fs_score = fisher_score.fisher_score(train_features, train_labels)

    idx = fisher_score.feature_ranking(fs_score)

    #print(idx)

    return idx

def cal_cr_balance_cr(prediction, y_val):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for index, (pred, y) in enumerate(zip(prediction, y_val)):

        if index < 10:
            if pred == y:
                TP += 1
            else:
                FP += 1
        else:
            if pred == y:
                TN += 1
            else:
                FN += 1

    CR = (TP+TN)/prediction.shape[0]

    TPR = TP/(TP + FN)
    TNR = TN/(FP + TN)

    balance_CR = (TPR + TNR)/2

    print([TP, FP, FN, TN])
    print(CR)
    print(balance_CR)

    return CR, balance_CR

def cal_acc(prediction, y_val):

    score = 0

    for index, (pred, y) in enumerate(zip(prediction, y_val)):
        if pred == y:
            score += 1

    acc = score/prediction.shape[0]

    return acc


def relative_power_lab(rp_ratio_array):
    
    rp_lab_array = []

    for band_index in range(0, 5):
        for channel in range(0, 30):
            for rec_channel in range(channel, 30):

                if rec_channel == channel:
                    continue
                else:
    
                    b_1 = rp_ratio_array[channel][band_index]
                    b_2 = rp_ratio_array[rec_channel][band_index]

                    rp_lab = (b_2 - b_1) / (b_1 + b_2)

                    rp_lab_array.append(rp_lab)

    rp_lab_array = np.asarray(rp_lab_array)

    return rp_lab_array

def lda(X_train, y_train, X_val, y_val):

    lda = LinearDiscriminantAnalysis()

    lda_object = lda.fit(X_train, y_train)

    prediction = lda.predict(X_val)

    fig, ax = plt.subplots()

    #Plot train data
    for index, train in enumerate(X_train):

        if index <13:
            #Patient Train data plot
            pt = ax.scatter(train[0], train[1], marker='o', c='r')
        else:
            #HC Train data plot
            ht = ax.scatter(train[0], train[1], marker='o', c='b')

    for index, val in enumerate(X_val):

        if index <10:
            #Patient Val data plot
            pv = ax.scatter(val[0], val[1], marker='x', c='r')
        else:
            #HC Val data plot
            hv = ax.scatter(val[0], val[1], marker='x', c='b')

    x1 = np.array([np.min(X_train[:,0], axis=0), np.max(X_train[:,0], axis=0)])

    #Plot line
    b, w1, w2 = lda.intercept_[0], lda.coef_[0][0], lda.coef_[0][1]
    y1 = -(b + x1*w1)/w2    
    ax.plot(x1, y1)

    ax.legend([pt, ht, pv, hv], ["Patient Train data", "HC Train data", "Patient Val data", "HC Val data"])

    plt.show()

    return prediction

def lda_loo(X_train, y_train, X_val, y_val):

    lda = LinearDiscriminantAnalysis()

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)

    lda_object = lda.fit(X_train, y_train)

    prediction = lda.predict(X_val)

    '''
    fig, ax = plt.subplots()

    #Plot train data
    for index, (X, y) in enumerate(zip(X_train, y_train)):

        if y == 1:
            #Patient Train data plot
            pt = ax.scatter(X[0], X[1], marker='o', c='r')
        else:
            #HC Train data plot
            ht = ax.scatter(X[0], X[1], marker='o', c='b')

    for index, val in enumerate(X_val):

        val = ax.scatter(val[0], val[1], marker='x', c='g')

    x1 = np.array([np.min(X_train[:,0], axis=0), np.max(X_train[:,0], axis=0)])

    #Plot line
    b, w1, w2 = lda.intercept_[0], lda.coef_[0][0], lda.coef_[0][1]
    y1 = -(b + x1*w1)/w2    
    ax.plot(x1, y1)

    ax.legend([pt, ht, val], ["Patient Train data", "HC Train data", "Val data"])

    plt.show()
    '''

    return prediction

def knn(X_train, y_train, X_val, y_val):

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    prediction = classifier.predict(X_val)

def main():
    
    mat = scipy.io.loadmat('Tee_170321.mat')

    data_raw = mat['data']

    train_X = []
    train_y = []
    val_X = []
    val_y = []

    idx = fisher(data_raw)

    first_feature = idx[0]
    second_feature = idx[1]

    for index, patient_subjects in enumerate(data_raw[0:23]):
    
        patient_features = compute(patient_subjects[0])

        if index < 13:
            train_X.append([patient_features[first_feature], patient_features[second_feature]])
            train_y.append(patient)
        else:
            val_X.append([patient_features[first_feature], patient_features[second_feature]])
            val_y.append(patient)

    for index, HC_subjects in enumerate(data_raw[23:47]):

        HC_features = compute(HC_subjects[0])

        if index < 14:
            train_X.append([HC_features[first_feature], HC_features[second_feature]])
            train_y.append(HC)
        else:
            val_X.append([HC_features[first_feature], HC_features[second_feature]])
            val_y.append(HC)

    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)
    val_X = np.asarray(val_X)
    val_y = np.asarray(val_y)

    print(train_X.shape)
    print(train_y.shape)
    print(val_X.shape)
    print(val_y.shape)

    com_pred = lda(train_X, train_y, val_X, val_y)
    #knn(train_X, train_y, val_X, val_y)

    cal_cr_balance_cr(com_pred, val_y)

def leave_one_out():

    mat = scipy.io.loadmat('Tee_170321.mat')

    data_raw = mat['data']

    X = []
    y = []

    idx = fisher(data_raw)

    first_feature = idx[0]
    second_feature = idx[1]

    with Bar('Processing') as bar:
        for index, subjects in enumerate(data_raw):
        
            features = compute(subjects[0])

            X.append([features[first_feature], features[second_feature]])

            if index < 23:
                y.append(patient)
            else:
                y.append(HC)

            bar.next()

    X = np.asarray(X)
    y = np.asarray(y)

    df = pd.DataFrame(X)

    df.to_csv("data.csv")

    loo = LeaveOneOut()

    com_pred = []

    for train_index, test_index in loo.split(X):

        train_X, val_X = X[train_index], X[test_index]
        train_y, val_y = y[train_index], y[test_index]

        pred = lda_loo(train_X, train_y, val_X, val_y)
        #pred = knn(train_X, train_y, val_X, val_y)

        com_pred.append(pred)

    com_pred = np.asarray(com_pred)

    acc = cal_acc(com_pred, y)

def add_one_feature():

    mat = scipy.io.loadmat('Tee_170321.mat')

    data_raw = mat['data']

    com_acc = []

    idx = fisher(data_raw)

    with Bar('Processing') as bar:

        for feature_index in range(1, 6):

            X = []
            y = []

            for index, subjects in enumerate(data_raw):
            
                features = compute(subjects[0])

                feature_list_index = [index for index in idx[0:feature_index]]

                X.append([features[i] for i in feature_list_index])

                if index < 23:
                    y.append(patient)
                else:
                    y.append(HC)

            X = np.asarray(X)
            y = np.asarray(y)

            df = pd.DataFrame(X)

            df.to_csv(str(feature_index) + ".csv")

            com_pred = []

            loo = LeaveOneOut()

            for train_index, test_index in loo.split(X):

                train_X, val_X = X[train_index], X[test_index]
                train_y, val_y = y[train_index], y[test_index]

                train_X = np.asarray(train_X)
                train_y = np.asarray(train_y)
                val_X = np.asarray(val_X)
                val_y = np.asarray(val_y)

                pred = lda_loo(train_X, train_y, val_X, val_y)
                #pred = knn(train_X, train_y, val_X, val_y)

                com_pred.append(pred)

            com_pred = np.asarray(com_pred)

            acc = cal_acc(com_pred, y)

            com_acc.append(acc)

            bar.next()

    com_acc = np.asarray(com_acc)

    print(com_acc)

    plt.plot(com_acc)
    plt.show()

def sequence_feature_selection():

    mat = scipy.io.loadmat('Tee_170321.mat')

    data_raw = mat['data']

    idx = fisher(data_raw)

    feature_list_index = [index for index in idx[0:5]]

    stack_features = []

    highest_acc = []

    with Bar('Processing') as bar:
        
        while len(stack_features) < 5:

            stack_y = []
            com_acc = []

            for com_feature_index in feature_list_index:

                com_fea = []
                X = []
                y = []

                for fea in stack_features:
                    com_fea.append(fea)
                    
                com_fea.append(com_feature_index)

                stack_y.append(com_fea)

                for index, subjects in enumerate(data_raw):
                
                    features = compute(subjects[0])

                    X.append([features[i] for i in com_fea])

                    if index < 23:
                        y.append(patient)
                    else:
                        y.append(HC)

                X = np.asarray(X)
                y = np.asarray(y)

                com_pred = []

                loo = LeaveOneOut()

                for train_index, test_index in loo.split(X):

                    train_X, val_X = X[train_index], X[test_index]
                    train_y, val_y = y[train_index], y[test_index]

                    train_X = np.asarray(train_X)
                    train_y = np.asarray(train_y)
                    val_X = np.asarray(val_X)
                    val_y = np.asarray(val_y)

                    pred = lda_loo(train_X, train_y, val_X, val_y)
                    #pred = knn(train_X, train_y, val_X, val_y)

                    com_pred.append(pred)

                com_pred = np.asarray(com_pred)

                acc = cal_acc(com_pred, y)

                com_acc.append(acc)

                bar.next()

            com_acc = np.asarray(com_acc)
            stack_y = np.asarray(stack_y)

            highest_index = np.argmax(com_acc)

            highest_acc.append(np.max(com_acc))

            stack_features = stack_y[highest_index]

            feature_list_index.remove(stack_features[-1])

    highest_acc = np.asarray(highest_acc)

    print(highest_acc)

    plt.plot(highest_acc)
    plt.show()

if __name__ == '__main__':
    #main()
    #leave_one_out()
    #add_one_feature()
    sequence_feature_selection()
    