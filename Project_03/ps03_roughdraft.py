from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np

data = load_iris().data[0:100, 0:2] #first 100 datas, considering the first 2 dimensions
target = load_iris().target[0:100]

train_data, test_data = train_test_split(data, test_size=0.8, random_state=0)
train_target, test_target = train_test_split(target, test_size=0.8, random_state=0)

# Fit the data to a logistic regression model
binary_lin_classifier = LogisticRegression(random_state=0).fit(train_data, train_target) #http://scikit-learn.org/stable/modules/linear_model.html

# Get Parameters
w1, w2 = binary_lin_classifier.coef_.T
b = binary_lin_classifier.intercept_[0]

'''
https://stats.stackexchange.com/questions/39243/how-does-one-interpret-svm-feature-weights
'''

# Intercept and Gradient of the decision boundary
# Weights = hyperplanethe; coordinates of a vector which is orthogonal to the hyperplane
c = -b/w2
m = -w1/w2

xmin, xmax = 2, 8
ymin, ymax = 1, 5
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')

plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.scatter(*train_data[train_target==0].T, s=8, alpha=0.5)
plt.scatter(*train_data[train_target==1].T, s=8, alpha=0.5)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')

#plt.show()
#plt.close()

# accuracy of your binary linear classifier
acc_train_score = accuracy_score(train_target, binary_lin_classifier.predict(train_data)) 
print('acc_train_score', acc_train_score)
acc_test_score = accuracy_score(test_target, binary_lin_classifier.predict(test_data)) 
print('acc_test_score', acc_test_score)



'''
3
'''
# SVM
SVM_clf = svm.SVC(kernel= 'linear', C=1000)
SVM_clf.fit(train_data, train_target)

# Get Parameters
w1, w2 = SVM_clf.coef_.T
b = SVM_clf.intercept_[0]

w = SVM_clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(4, 7)
yy = a * xx - (SVM_clf.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(SVM_clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.plot(xx, yy, "k-")
plt.plot(xx, yy_down, "k-")
plt.plot(xx, yy_up, "k-")

plt.scatter(SVM_clf.support_vectors_[:, 0], SVM_clf.support_vectors_[:, 1], s=80,
 facecolors="none", zorder=10, edgecolors="k")
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, zorder=10, cmap=plt.cm.Paired,
 edgecolors="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Accuracy
acc_train_score = accuracy_score(train_target, SVM_clf.predict(train_data)) 
print('SVM_acc_train_score', acc_train_score)
acc_test_score = accuracy_score(test_target, SVM_clf.predict(test_data)) 
print('SVM_acc_test_score', acc_test_score)

print('margin', margin)

'''
8
'''
data = load_iris().data[0:100, 0:2] #first 100 datas, considering the first 2 dimensions
target = load_iris().target[0:100]

train_data, test_data = train_test_split(data, test_size=0.4, random_state=0)
train_target, test_target = train_test_split(target, test_size=0.4, random_state=0)

SVM_clf = svm.SVC(kernel= 'linear', C=1000)
SVM_clf.fit(train_data, train_target)

# Get Parameters
w1, w2 = SVM_clf.coef_.T
b = SVM_clf.intercept_[0]
print('w1', w1)
print('w2', w2)
print('b', b)

w = SVM_clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(4, 7)
yy = a * xx - (SVM_clf.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(SVM_clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.plot(xx, yy, "k-")
plt.plot(xx, yy_down, "k-")
plt.plot(xx, yy_up, "k-")

plt.scatter(SVM_clf.support_vectors_[:, 0], SVM_clf.support_vectors_[:, 1], s=80,
 facecolors="none", zorder=10, edgecolors="k")
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, zorder=10, cmap=plt.cm.Paired,
 edgecolors="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Accuracy
acc_train_score = accuracy_score(train_target, SVM_clf.predict(train_data)) 
print('SVM_acc_train_score', acc_train_score)
acc_test_score = accuracy_score(test_target, SVM_clf.predict(test_data)) 
print('SVM_acc_test_score', acc_test_score)

print('margin', margin)

