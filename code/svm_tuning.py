from sklearn import svm
from sys import argv
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import numpy

cpu_count = 1
if(len(argv) == 2):
	script, cpu_count = argv
if !isinstance(cpu_count, (int, long)):
	print "Cpu count should be integer"
	exit()
print cpu_count
exit()
dataset = numpy.genfromtxt(open('../Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
target = [x[0] for x in dataset]
target = numpy.array(target)
train = [x[1:] for x in dataset]
train = numpy.array(train)
#test = numpy.genfromtxt(open('../Data/test.csv','r'), delimiter=',', dtype='f8')[1:]
number_of_folds = 10
k_fold = KFold(n_splits = number_of_folds)
mean_accuracy = 0
experiments = [
	[50, 0.1, 1e-10],
	[60, 0.01, 1e-10],
	[50, 1, 1e-8],
	[50, 0.01, 1e-6],
	[40, 0.1, 1e-8],
	[40, 0.01, 1e-9],
	[60, 0.1, 1e-10],
	[70, 0.1, 1e-7],
	[80, 0.1, 1e-7],
	[30, 0.01, 1e-9],
	[30, 0.1, 1e-7],
	[40, 0.1, 1e-10],
	[70, 0.01, 1e-9],
	[60, 0.1, 1e-8],
	[30, 0.1, 1e-10],
]
accuracies = []
experiment_number = 1
for experiment in experiments:
	print "Experiment: %d " % experiment_number
	cm_file = open("../Confusion Matrix/SVM/Exp_%d.txt" % experiment_number, 'w')
	k = 1
	number_of_svms = experiment[0]
	for train_index, test_index in k_fold.split(train):
		fold_train = train[train_index]
		fold_test = train[test_index]
		fold_target_train = target[train_index]
		fold_target_test = target[test_index]
		svm_bagging_classifier = OneVsRestClassifier(BaggingClassifier(svm.SVC(C = experiment[1], gamma = experiment[2]), max_samples = 1.0 / number_of_svms, n_estimators = number_of_svms, n_jobs = cpu_count))
		svm_bagging_classifier.fit(fold_train, fold_target_train)
		predictions = svm_bagging_classifier.predict(fold_test)
		mean_accuracy += accuracy_score(fold_target_test, predictions)
		fold_cm = confusion_matrix(fold_target_test, predictions)
		cm_file.write('Fold - %d\n%s\n' % (k, fold_cm))
		k += 1
	mean_accuracy /= number_of_folds
	print "Accuracy: %d" % mean_accuracy
	accuracies.append(mean_accuracy)
	cm_file.close()
	experiment_number += 1
print "\n %s" % accuracies
#savetxt('svm_predictions.csv', numpy.c_[range(1,len(test)+1), predictions], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')