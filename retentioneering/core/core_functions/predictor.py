from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import shap


DEFAULT_PARAMS = {'max_depth': 8,
                  'objective': 'binary:logistic',
                  'nthread': 4,
                  'eta': 0.5,
                  'eval_metric': 'auc',
                  'min_child_weight': 8,
                  'lambda': 6,
                  'subsample': 0.5
                 }


class Predictor:
    def __init__(self, params=DEFAULT_PARAMS):
        self.params = params

    def train(self, features, targets):
        features['target'] = features.index.isin(targets)

        X = features.drop(columns=['target'])
        y = features['target'].map(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        evallist = [(dtest, 'eval')]

        num_round = 9999
        self.model = xgb.train(self.params,
                               dtrain,
                               num_round,
                               evallist,
                               early_stopping_rounds=16)

    def export_model(self, file_name):
        pickle.dump(self.model, open(f"{file_name}.pkl", "wb"))

    def import_model(self, file_name):
        self.model = pickle.load(open(f"{file_name}.pkl", "rb"))

    def score(self, features, targets):
        features['target'] = features.index.isin(targets)

        X = features.drop(columns=['target'])
        y = features['target'].map(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        y_score = self.model.predict(dtest)
        fpr, tpr, thresh_ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        y_score_train = self.model.predict(dtrain)
        fpr_tr, tpr_tr, _ = roc_curve(y_train, y_score_train)
        roc_auc_tr = auc(fpr_tr, tpr_tr)

        sns.set(style='white', context='notebook', rc={'figure.figsize': (8, 5)})

        plt.figure()
        lw = 2

        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve test (area = %0.3f)' % roc_auc)

        plt.plot(fpr_tr, tpr_tr, color='darkgreen',
                 lw=lw, label='ROC curve train (area = %0.3f)' % roc_auc_tr)

        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='random guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()

    def explain(self, features, targets):
        features['target'] = features.index.isin(targets)

        X = features.drop(columns=['target'])
        y = features['target'].map(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        shap.initjs()

        mybooster = self.model
        model_bytearray = mybooster.save_raw()[4:]
        def temp(self=None):
            return model_bytearray
        mybooster.save_raw = temp

        explainer = shap.TreeExplainer(mybooster)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train)

