import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def test_result(model, test_data_loader: DataLoader, criterion: object, metric: object):
    '''
    Функция принимает на вход модель и тестовый датасет и вычисляет по нему значение метрики и предсказанные 
    значения
    Входные данные:
    model - обученная модель
    test_data_loader: DataLoader - тестовый датасет, приведенный к формату DataLoader
    criterion - критерий оценки потерь модели
    metric - метрика оценки модели
    '''
    model.eval()
    test_metrics = []
    test_losses = []
    result = {}
    y_true = []
    y_proba = []
    for batch_idx, data in enumerate(test_data_loader):
        inputs = data[0]
        target = data[1]
        outputs = model(inputs)        
        loss = criterion(outputs[:, 0], target.to(torch.float32))
        test_losses.append(loss.item())
        y_true.append(np.array(target.detach().numpy()))
        y_proba.append(np.array(outputs[:,0].detach().numpy()))

    y_true = np.concatenate(np.array(y_true), axis=0)
    y_proba = np.concatenate(np.array(y_proba), axis=0)

    test_loss    = np.mean(test_losses)
    test_metric  = metric(y_true, y_proba)
    result['test_loss'] = test_loss
    result['test_metric'] = test_metric
    print(result)
    return y_true, y_proba

def show_proba_calibration_plots(y_predicted_probs, y_true_labels):
    '''
    Функция принимает на вход предсказзные моделью значения и истинные метки и строит изменения f1, precision, 
    recall при изменяющемся значении порога от 0.1 до 0.9
    Входные данные:
    y_predicted_probs - предсказанные моделью значения
    y_true_labels - истинные значения
    '''
    preds_with_true_labels = np.array(list(zip(y_predicted_probs, y_true_labels)))

    thresholds = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in np.linspace(0.1, 0.9, 9):
        thresholds.append(threshold)
        precisions.append(precision_score(y_true_labels, list(map(int, y_predicted_probs > threshold))))
        recalls.append(recall_score(y_true_labels, list(map(int, y_predicted_probs > threshold))))
        f1_scores.append(f1_score(y_true_labels, list(map(int, y_predicted_probs > threshold))))

    scores_table = pd.DataFrame({'f1':f1_scores,
                                 'precision':precisions,
                                 'recall':recalls,
                                 'probability':thresholds}).sort_values('f1', ascending=False).round(3)
  
    figure = plt.figure(figsize = (15, 5))

    plt1 = figure.add_subplot(121)
    plt1.plot(thresholds, precisions, label='Precision', linewidth=4)
    plt1.plot(thresholds, recalls, label='Recall', linewidth=4)
    plt1.plot(thresholds, f1_scores, label='F1', linewidth=4)
    plt1.set_ylabel('Scores')
    plt1.set_xlabel('Probability threshold')
    plt1.set_title('Probabilities threshold calibration')
    plt1.legend(bbox_to_anchor=(0.25, 0.25))   
    plt1.table(cellText = scores_table.values,
               colLabels = scores_table.columns, 
               colLoc = 'center', cellLoc = 'center', loc = 'bottom', bbox = [0, -1.3, 1, 1])

    plt2 = figure.add_subplot(122)
    plt2.hist(preds_with_true_labels[preds_with_true_labels[:, 1] == 0][:, 0], 
              label='Сlass 0', color='royalblue', alpha=1)
    plt2.hist(preds_with_true_labels[preds_with_true_labels[:, 1] == 1][:, 0], 
              label='Class 1', color='darkcyan', alpha=0.8)
    plt2.set_ylabel('Number of examples')
    plt2.set_xlabel('Probabilities')
    plt2.set_title('Probability histogram')
    plt2.legend(bbox_to_anchor=(1, 1))

    plt.show()

def labels_pred(pred_proba, threshold):
    '''
    Функция принимает на вход предсказзные моделью значения и значение порога и преобразует значения
    к бинарной классификации
    Входные данные:
    pred_proba - предсказанные моделью значения
    threshold - пороговое значение
    '''
    _labels_pred = np.array(list(map(lambda x: 0 if x<threshold else 1, pred_proba)))
    return _labels_pred

def report(y_true, y_prob, threshold=0.4):
    '''
    Функция принимает на вход предсказзные моделью значения, истинные значения и значение порога
    и получает матрицу ошибок для заданного порога
    Входные данные:
    y_true - истинные значения
    y_prob - предсказанные моделью значения
    threshold=0.4 - пороговое значение
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, labels_pred(y_prob, threshold)).ravel()
    print(classification_report(y_true, labels_pred(y_prob, threshold)))
    print('---------------')
    print('Матрица ошибок:')
    print(confusion_matrix(y_true, labels_pred(y_prob, threshold)))
    print('---------------')
    print(f'Правильно предсказано 0: {tn}')
    print(f'Правильно предсказано 1: {tp}')
    print(f'Ложноотрицательных (FN, ош2р, пропуск события): {fn}')
    print(f'Ложноположительных (FP, ош1р, ложная  тревога): {fp}')
    print(f'Всего ошибок: {fp+fn}')

def plot_precision_recall(y_true, y_proba):
    '''
    Функция принимает на вход предсказаные моделью значения и истинные значения и строит PR кривую с f1 изолиниями
    Входные данные:
    y_true - истинные значения
    y_prob - предсказанные моделью значения
    '''
    _, ax = plt.subplots(figsize=(7, 7))

    prec, recall, _ = precision_recall_curve(y_true, y_proba)
    ax.plot(recall, prec, linewidth=2, color='r')

    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([min(prec), 1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid()

def plot_PR_ROC(y_true, y_proba):
    '''
    Функция принимает на вход предсказаные моделью значения и истинные значения и строит PR и ROC кривые
    Входные данные:
    y_true - истинные значения
    y_prob - предсказанные моделью значения
    '''
    figure = plt.figure(figsize = (16, 7))
    plt1 = figure.add_subplot(121)

    prec, recall, _ = precision_recall_curve(y_true, y_proba)
    plt1.plot(recall, prec, linewidth=2, color='r')

    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt1.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.5)

    plt1.set_xlim([0.0, 1.0])
    plt1.set_ylim([min(prec), 1.01])
    plt1.set_xlabel('Recall')
    plt1.set_ylabel('Precision')
    plt1.grid()

    plt2 = figure.add_subplot(122)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt2.set_xlim([0.0, 1.0])
    plt2.set_ylim([0.0, 1.0])
    plt2.plot(fpr, tpr)
    plt2.set_xlabel('False positive rate')
    plt2.set_ylabel('True positive rate')
    plt2.set_title(f'ROC curve. ROC_AUC = {roc_auc:.3f}')
    plt2.grid()
    plt.show()

