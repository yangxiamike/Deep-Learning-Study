from linear_model.LogisticRegression import Logistic_Regression
from sklearn.model_selection import StratifiedKFold
import time
from datasets.load_balanced import load_balanced
from sklearn.model_selection import KFold,LeaveOneOut
from sklearn.metrics import roc_curve, auc,accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from scipy import interp


"""
This document contains all the codes that output the experiment results
The self-implemented logistic regression and data generator are saved in folders.

datasets and linear_model
"""
"""
Compare self-implemented algorithm with Sklearn
"""
#--------------------------------------------------
#Ten Fold Cross Validation
#--------------------------------------------------

# #############################################################################
# Data IO and generation

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=10)
# classifier = Logistic_Regression(random_state=0)
classifier = LogisticRegression(random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    # classifier.fit(X[train], y[train],alpha = 0.05,max_iter=1000,step_size=0.1)
    # probas_ = classifier.predict_prob(X[test])
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve

    # fpr, tpr, thresholds = roc_curve(y[test], probas_)
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Sklearn Logistic Regression')
# plt.title('Implementation of Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('iris_logi_skr.png')
plt.show()


#--------------------------------------------------
#Leave One Out
#--------------------------------------------------
"""
Sklearn
"""
time1 = time.clock()
loo = LeaveOneOut()
scores = []
for train, test in loo.split(X):
    model = LogisticRegression(random_state = 1)
    model.fit(X[train],y[train])
    y_pre = model.predict(X[test])
    score = accuracy_score(y[test],y_pre)
    scores.append(score)
time2 = time.clock()
print('average precision %f' %np.array(scores).mean())
print('%fs is used ' %round(time2-time1,2))

"""
Self-implemented 
"""
time1 = time.clock()
loo = LeaveOneOut()
scores = []
for train, test in loo.split(X):
    model = Logistic_Regression(random_state = 1)
    model.fit(X[train],y[train])
    y_pre = model.predict(X[test])
    score = accuracy_score(y[test],y_pre)
    scores.append(score)
time2 = time.clock()
print('average precision %f' %np.array(scores).mean())
print('%fs is used ' %round(time2-time1,2))


#--------------------------------------------------
#Ten Fold Cross Validation
#--------------------------------------------------

# #############################################################################
# Data IO and generation
scores1 = []
# Import some data to play with
X, y = load_balanced()
X, y = X[y != 0], y[y != 0]
y[y == 2] = 0
n_samples, n_features = X.shape

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=10)
classifier = Logistic_Regression(random_state=0)
# classifier = LogisticRegression(random_state = 0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    classifier.fit(X[train], y[train], alpha=0.05, max_iter=1000, step_size=0.1)
    probas_ = classifier.predict_prob(X[test])
    # probas_ = classifier.fit(X[train],y[train]).predict_proba(X[test])

    scores1.append(accuracy_score(y[test], classifier.predict(X[test])))

    # Compute ROC curve and area the curve

    fpr, tpr, thresholds = roc_curve(y[test], probas_)
    # fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Sklearn Logistic Regression')
plt.title('Implementation of Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('balanced_logi_self.png')
plt.show()


"""
Compare Softmax with OvO and OvR
"""

#--------------------------------------------------
#Softmax Iris
#--------------------------------------------------
# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output


# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1,
                                                    random_state=0)

test = y_test
train = y_train

# Learn to predict each class against the other
#classifier = OneVsRestClassifier(LogisticRegression(random_state=0))
#classifier = LogisticRegression(multi_class='multinomial',solver = 'newton-cg')

# softmax
softmax = LogisticRegression(multi_class='multinomial',solver='newton-cg')
#y_soft = softmax.fit(X_train,y_train).decision_function(X_test)

# onevsone
logi_ovo = OneVsOneClassifier(LogisticRegression())
#logi_ovo.fit(X_train,y_train)
#y_ovo = softmax.fit(X_train,y_train).decision_function(X_test)

# onevsall
logi_ovr = OneVsRestClassifier(LogisticRegression())
#y_ovr = logi_ovr.fit(X_train,y_train).decision_function(X_test)

y_score = softmax.fit(X_train, y_train).decision_function(X_test)

y_test = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"],
         label='Softmax ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
#------------------------------
y_score = logi_ovr.fit(X_train, y_train).decision_function(X_test)

y_test = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"],
         label='OvR ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='darkorange', linestyle='--', linewidth=4)

#---------------------
y_score = logi_ovo.fit(X_train, y_train).decision_function(X_test)

y_test = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"],
         label='OvO ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='red', linestyle='--', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison')
plt.legend(loc="lower right")
plt.savefig('iris_soft.png')
plt.show()

#--------------------------------------------------
#Softmax Balance
#--------------------------------------------------

# Import some data to play with
X,y = load_balanced()

# Binarize the output


# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15,
                                                    random_state=0)
test = y_test
train = y_train

# Learn to predict each class against the other
#classifier = OneVsRestClassifier(LogisticRegression(random_state=0))
#classifier = LogisticRegression(multi_class='multinomial',solver = 'newton-cg')

# softmax
softmax = LogisticRegression(multi_class='multinomial',solver='newton-cg',class_weight='balanced')
#y_soft = softmax.fit(X_train,y_train).decision_function(X_test)

# onevsone
logi_ovo = OneVsOneClassifier(LogisticRegression())
#logi_ovo.fit(X_train,y_train)
#y_ovo = softmax.fit(X_train,y_train).decision_function(X_test)

# onevsall
logi_ovr = OneVsRestClassifier(LogisticRegression())
#y_ovr = logi_ovr.fit(X_train,y_train).decision_function(X_test)

y_score = softmax.fit(X_train, y_train).decision_function(X_test)

y_test = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"],
         label='Softmax ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
#------------------------------
y_score = logi_ovr.fit(X_train, y_train).decision_function(X_test)

y_test = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"],
         label='OvR ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='darkorange', linestyle='--', linewidth=4)

#---------------------
y_score = logi_ovo.fit(X_train, y_train).decision_function(X_test)

y_test = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"],
         label='OvO ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='red', linestyle='--', linewidth=2)
lw = 2
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison')
plt.legend(loc="lower right")
plt.savefig('balanced_soft.png')
plt.show()

acc1 = accuracy_score(test,softmax.predict(X_test))
acc2 = accuracy_score(test,logi_ovo.predict(X_test))
acc3 = accuracy_score(test,logi_ovr.predict(X_test))
print(acc1,acc2,acc3)