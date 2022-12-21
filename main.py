import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from lightgbm import LGBMClassifier
from utils import log_config


def preprocess_data(file_path, phase='train'):
    data = pd.read_csv(file_path).set_index('id', drop=True)
    # convert string to number
    lb = LabelEncoder()
    for col in data.columns:
        if data.loc[:, col].dtypes == 'object':
            # val, _ = pd.factorize(data.loc[:, col])
            # data.loc[:, col] = val
            data.loc[:, col] = lb.fit_transform(data[col])
    if phase == 'train':
        feature = data.iloc[:, :-1]
        label = data.iloc[:, -1]
        return feature, label
    elif phase == 'test':
        feature = data
        return feature


def create_model(model):
    if model == 'SVC':
        return SVC()
    elif model == 'KNeighborsClassifier':
        return KNeighborsClassifier(n_neighbors=40)
    elif model == 'RandomForestClassifier':
        return RandomForestClassifier(n_estimators=150)
    elif model == 'DecisionTreeClassifier':
        return DecisionTreeClassifier()
    elif model == 'AdaBoostClassifier':
        return AdaBoostClassifier()
    elif model == 'GradientBoostingClassifier':
        return GradientBoostingClassifier(n_estimators=200)
    elif model == 'MLPClassifier':
        return MLPClassifier()
    elif model == 'RidgeClassifier':
        return RidgeClassifier()
    elif model == 'SGDClassifier':
        return SGDClassifier()
    elif model == 'LGBMClassifier':
        return LGBMClassifier(n_estimators=525, learning_rate=0.01)
    elif model == 'VotingClassifier':
        return VotingClassifier(estimators=[
            ('RandomForestClassifier', RandomForestClassifier(n_estimators=150)),
            ('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=200)),
            ('LGBMClassifier', LGBMClassifier(n_estimators=525, learning_rate=0.01)),
        ], weights=[1, 0.8, 1.5])
    else:
        raise Exception('Invalid model name.')


def main():
    model_name = 'LGBMClassifier'

    train_path = 'train.csv'
    test_path = 'test.csv'
    submission_path = 'submission.csv'
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    log_path = 'log.txt'
    log = log_config(log_path)
    log.info(f'model: {model_name}')

    train_feature, train_label = preprocess_data(train_path, phase='train')

    # train
    model = create_model(model_name)
    model.fit(train_feature, train_label)
    scores = cross_val_score(model, train_feature, train_label, cv=5)
    score = np.mean(scores)
    log.info(f'score: {score}')

    if model == 'LGBMClassifier':
        from lightgbm import plot_importance
        import matplotlib.pyplot as plt
        plot_importance(model, figsize=(12, 8))
        plt.savefig('feature_importance.png')
        plt.show()

    # save model
    model_save_path = os.path.join(model_dir, f'{model_name}_{score:.4f}.dat')
    joblib.dump(model, model_save_path)

    # predict
    test_feature = preprocess_data(test_path, phase='test')
    submission = pd.read_csv(submission_path)
    submission.iloc[:, 1] = model.predict(test_feature)
    submission.iloc[:, 1] = submission.iloc[:, 1].map({1: 'yes', 0: 'no'})
    submission.to_csv(submission_path, index=False)
    value_counts = submission['subscribe'].value_counts()
    log.info(f'predict results:\n{value_counts}')


if __name__ == '__main__':
    main()
