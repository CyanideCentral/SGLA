from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from munkres import Munkres
import config as config

class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        
        if numclass1 != numclass2:
            print('Class Not equal!!!!')
            missing_classes = set(l1)-set(l2)
            for i, c in enumerate(missing_classes):
                self.pred_label[i] = c
                l2.append(c)
            numclass2 = len(l2)
        cost = np.zeros((numclass1, numclass2), dtype=np.float64)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]

            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        return acc, f1_macro, precision_macro, recall_macro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1, pre, rc = self.clusteringAcc()

        return acc, nmi, f1, pre, adjscore, rc

def ovr_evaluate(embeds, labels):
    # Labeled data from 10% to 90%
    if config.scale:
        labeled = np.arange(0.9, 1.0, 0.01)[::-1]
    else:
        labeled = np.arange(0.1, 1.0, 0.1)[::-1]
    for test_ratio in labeled:
        f1macros = []
        f1micros = []
        roc_auc_macros = []
        roc_auc_micros = []
        # Average over five random splits
        for seed in range(5):
            train_x, test_x, train_y, test_y = train_test_split(embeds, labels, test_size=test_ratio, random_state=seed)
            ovr_classifier = OneVsRestClassifier(LogisticRegression())
            ovr_classifier.fit(train_x, train_y)
            pred_labels = ovr_classifier.predict(test_x)
            pred_probs = ovr_classifier.predict_proba(test_x)
            if len(set(test_y)) == 2:
                pred_probs = pred_probs[:,1]
            f1macros.append(metrics.f1_score(test_y, pred_labels, average='macro'))
            f1micros.append(metrics.f1_score(test_y, pred_labels, average='micro'))
            roc_auc_macros.append(metrics.roc_auc_score(test_y, pred_probs, average='macro', multi_class='ovr'))
            roc_auc_micros.append(metrics.roc_auc_score(test_y, pred_probs, average='micro', multi_class='ovr'))
        print(f'Labeled data {int(100-test_ratio*100)}%: {np.mean(f1macros)} {np.mean(f1micros)} {np.mean(roc_auc_macros)} {np.mean(roc_auc_micros)}')
        print(f'Labeled data {int(100-test_ratio*100)}%: f1_macro: {np.mean(f1macros):.3f}, f1_micro: {np.mean(f1micros):.3f}, roc_auc_macro: {np.mean(roc_auc_macros):.3f}, roc_auc_micro: {np.mean(roc_auc_micros):.3f}')
    return np.mean(f1macros), np.mean(f1micros), np.mean(roc_auc_macros), np.mean(roc_auc_micros)