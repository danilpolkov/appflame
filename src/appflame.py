import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
import tensorflow as tf
import datetime
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def score_models(models,
                 X_train,
                 y_train,
                 CV_FOLDS,
                 accuracy_results,
                 time_to_fit,
                 names,
                 name_suffix=''):
    
    '''
    score list of models with cross validation
    
    return:
    ---------
    list with values
        score values for every fold
        time that takes to evaluate every fold
        name of the model
    
    '''
    
    for name, model in models:
        kfold = KFold(n_splits=CV_FOLDS)
        cv_results = cross_validate(model,
                                    X_train,
                                    y_train.values.ravel(),
                                    cv=kfold,
                                    scoring='roc_auc')
        accuracy_results.append(cv_results['test_score'])
        time_to_fit.append(cv_results['fit_time'])
        names.append(name + name_suffix)
        msg = '%s: roc-auc mean: %f std: (%f), takes: %f' % (name, cv_results['test_score'].mean(), 
                                                             cv_results['test_score'].std(), 
                                                             cv_results['fit_time'].mean())
        print(msg)
    return [accuracy_results, time_to_fit, names]


class Neural_Network:
    
    '''
    build neural netwotk 
    
    '''

    def __init__(self):
        self.epoch = 1
        self.metrics = ['AUC']
        self.model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(64, input_dim = 32, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])  
        self.model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[self.metrics])
        
    def epoch(self, n):
        self.epoch = n
        return self.epoch
    
    def fit(self, X, y):
        self.model.fit(X, y, batch_size = 32, epochs=self.epoch)
        return self.model
        
    def predict(self, X):
        return self.model.predict_classes(X)
    
    def predict_score(self, X):
        return self.model.predict(X)
    
    
    
    
def train_nn_on_fold(model, x_train_nn, y_train, i, rows_per_fold):
    
    '''
    train neural network on several folds
    
    return:
    --------
    list with values
        list of scores for every fold
        list of time that takes to evaluate every fold
        
    '''

    start = datetime.datetime.now()
    train = x_train_nn.drop(range(rows_per_fold * i, rows_per_fold * (i + 1)))
    test = y_train.drop(range(rows_per_fold * i, rows_per_fold * (i + 1)))
    model.fit(train, test)
    prediction = model.predict_score(
        x_train_nn[rows_per_fold * i:rows_per_fold * (i + 1)].values)
    score = roc_auc_score(y_train[rows_per_fold * i:rows_per_fold * (i + 1)],
                          prediction)
    end = datetime.datetime.now()
    time = (end - start).seconds
    return score, time
   

def check_confusion_matrix(model, X_train, X_test, y_train, y_test, name):
    
    '''
    train model on full test dataset and make prediction for test
    show classification report and plot confusion matrix
    
    '''
    
    model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_test)

    print(classification_report(y_test, pred))

    confmat = confusion_matrix(y_true=y_test, y_pred=pred)
    classes = ['0', '1']
    df_cm = pd.DataFrame(confmat, columns=classes, index=classes)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(12, 9))

    ax = plt.axes()
    sns.heatmap(df_cm,
                cmap="Blues",
                annot=True,
                annot_kws={"size": 15},
                fmt="d")
    sns.set(font_scale=0.9)
    ax.set_title('Confusion Matrix for ' + name, fontsize=20)
    plt.show()
    

def show_comparing_table(names, accuracy_results, time_to_fit):
    
    '''
    print comparing table with model name, avg score and time to fit
    
    '''
    
    comparing_df = pd.DataFrame(np.array([names, accuracy_results, time_to_fit]).T,
                                columns=['name', 'roc-auc', 'time to learn'])
    
    comparing_df['roc-auc'] = comparing_df['roc-auc'].apply(lambda x: np.mean(x))
    comparing_df['time to learn'] = comparing_df['time to learn'].apply(lambda x: np.mean(x))
    print(comparing_df)
