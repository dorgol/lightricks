import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_validate, cross_val_predict

import basic


class PredictingModel:
    def __init__(self, x, y, model):
        self.x = x
        self.y = y
        self.model = model

    def fitting_model(self, **kwargs):
        model_to_fit = self.model(**kwargs)
        return model_to_fit.fit(self.x, self.y)

    @staticmethod
    def predict_report(fitted, x_test: pd.DataFrame, y_test):
        """
        static method for prediction report. The report includes accuracy score, confusion matrix and a general report
        :param fitted: fitted model
        :param x_test: unseen features data
        :param y_test: unseen dependent variable
        :return: see above
        """
        y_pred = fitted.predict(x_test)
        ac_score = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return y_pred, ac_score, confusion, report

    def fit_cv(self, **kwargs) -> dict:
        """
        fit cross validation for the instance model
        :param kwargs: optional arguments
        :return: dictionary of results for every
        """
        cv_result = cross_validate(self.model, self.x, self.y, **kwargs)
        return cv_result

    @staticmethod
    def create_report(model_art, model_name):
        """
        report the desired scoring for a given model
        :param model_art: fitted cv model
        :param model_name: arbitrary model name
        :return: dictionary of outcomes
        """
        keys = [key for key, value in model_art.items() if key.startswith("test")]
        values = [model_art[keys[i]].mean() for i in range(len(keys))]
        values.append(model_name)
        keys.append('model_name')
        results = pd.DataFrame([values], columns=keys)
        return results.set_index(['model_name'])

    def create_confusion(self):
        """
        creates confusion matrix for cross validation
        :return: see above
        """
        y_pred = cross_val_predict(self.model, self.x, self.y, cv=5)
        return confusion_matrix(self.y, y_pred), classification_report(self.y, y_pred)


def running_models(x, y):
    """
    this function trains multiple models and summarizes the results in a dataframe
    :param x: features, result of basic.piping
    :param y: target, result of basic.piping
    :return: dataframe of results
    """
    lr = PredictingModel(x, y, LogisticRegression())
    lr_fit = lr.fit_cv(scoring=['balanced_accuracy', 'f1'])

    rf = PredictingModel(x, y, RandomForestClassifier())
    rf_fit = rf.fit_cv(scoring=['balanced_accuracy', 'f1'], n_jobs=2)

    lr_balanced = PredictingModel(x, y, LogisticRegression(class_weight="balanced"))
    lr_balanced_fit = lr_balanced.fit_cv(scoring=['balanced_accuracy', 'f1'])

    rf_balanced = PredictingModel(x, y, RandomForestClassifier(class_weight="balanced"))
    rf_balanced_fit = rf_balanced.fit_cv(scoring=['balanced_accuracy', 'f1'], n_jobs=2)

    lr_under = PredictingModel(x, y, Pipeline([('rs', RandomUnderSampler()), ('lr', LogisticRegression())]))
    lr_under_fit = lr_under.fit_cv(scoring=['balanced_accuracy', 'f1'])

    rf_under = PredictingModel(x, y, Pipeline([('rs', RandomUnderSampler()), ('lr', RandomForestClassifier())]))
    rf_under_fit = rf_under.fit_cv(scoring=['balanced_accuracy', 'f1'])

    lr_smote = PredictingModel(x, y, Pipeline([('smote', SMOTE()), ('lr', LogisticRegression())]))
    lr_smote_fit = lr_smote.fit_cv(scoring=['balanced_accuracy', 'f1'])

    rf_smote = PredictingModel(x, y, Pipeline([('smote', SMOTE()), ('lr', RandomForestClassifier())]))
    rf_smote_fit = rf_smote.fit_cv(scoring=['balanced_accuracy', 'f1'])

    models_list = [lr_fit, rf_fit, lr_balanced_fit, rf_balanced_fit, lr_under_fit, rf_under_fit,
                   lr_smote_fit, rf_smote_fit]
    models_name = ["logistic_regression", "random_forest", "logistic_regression_balanced", "random_forest_balanced",
                   "logistic_regression_undersampling", "random_forest_undersampling",
                   "logistic_regression_smote", "random_forest_smote"]
    lis = []
    for i in range(len(models_list)):
        report = PredictingModel.create_report(models_list[i], models_name[i])
        lis.append(report)
    report = pd.concat(lis)
    return report


def full_cycle_model():
    """
    function to create a logistic regression with holdout data.
    :return: prediction report
    """
    data = basic.piping('additive')
    data = basic.scaling(data)
    x_train, x_test, y_train, y_test = basic.split(data, 'train')
    logit = PredictingModel(x_train, y_train, LogisticRegression)
    logit = logit.fitting_model()
    return PredictingModel.predict_report(logit, x_test, y_test)


_, _, confusion_basic_log, report_basic_log = full_cycle_model()
