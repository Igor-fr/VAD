import os
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
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
import matplotlib.pyplot as plt

def create_spec_dataset(speech_labels_paths: str, path_specs: str, delimiter: str = ','):
    '''
    Функция принимает на вход пути ко всем аудио датасета и преобразует их к спектрограммам
    Входные данные:
    speech_labels_paths: str - путь к датасету AVA формата
    path_specs: str - путь к папке, куда сохранять спектрограммы
    delimeter: str=',' - разделитель в датасете
    '''
    with open(speech_labels_paths) as f:
        speech_labels = f.readlines()
    speech_labels = [x.replace('\n', '') for x in speech_labels]

    if not os.path.exists(path_specs):
        os.makedirs(path_specs)

    i = 0
    labels = []
    sample_rate = 16000
    win_length = hop_length = n_fft = int(sample_rate / 100)
    step = int(0.32 * sample_rate)
    transformer = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, win_length=win_length,
                                                                n_mels=32, power=1, hop_length=hop_length,
                                                                n_fft=n_fft + 1)
    while i < len(speech_labels):
        path_to_audio = speech_labels[i].split(delimiter)[0]
        audio = path_to_audio.split('/')[-1].split('.')[0]
        try:
            waveform, _ = torchaudio.load(path_to_audio)
            print(path_to_audio)
        except:
            i += 1
            continue
        j = 0
        while True:

            speech_label_split = speech_labels[i].split(delimiter)
            second_start = float(speech_label_split[1])
            second_end = float(speech_label_split[2])
            label = speech_label_split[3]
            if label == 'NO_SPEECH':
                label = '0'
            else:
                label = '1'
            while (second_start * sample_rate < waveform.shape[1] - step) and (second_end - second_start > 0.32):
                specgram = transformer(waveform[0][int(second_start * sample_rate):int(second_start * sample_rate) + step])
                data = specgram.log2().detach().numpy()
                if True in np.isinf(data):
                    j += 1
                    second_start += 0.32
                    continue
                np.save(path_specs + '/{0}_{1}'.format(audio, str(j)), data)
                labels.append([path_specs + '/{0}_{1}.npy'.format(audio, str(j)), label])
                j += 1
                second_start += 0.32
            i += 1
            if i == len(speech_labels):
                break
            if speech_labels[i].split(delimiter)[0] != path_to_audio:
                break
    return np.array(labels)

def create_train_test_dataset(data: str, test_size: float=0.2):
    '''
    Функция принимает на вход файл с указанием путей ко всем спектрограммам и их меткам
    Входные данные:
    data: str - список путей к спектрограммам и их меток
    test_size: float=0.2 - доля тестовой выборки в общем количестве данных
    '''
    np.random.shuffle(data)
    df = pd.DataFrame(data)
    df = df.rename(columns={0: 'spec_paths', 1: 'labels'})
    m = df.loc[:, 'labels'] == '0'
    df.loc[m, 'labels'] = 0
    m = df.loc[:, 'labels'] != 0
    df.loc[m, 'labels'] = 1
    test_df = df.loc[:int(df.shape[0]*test_size)].reset_index()
    train_df = df.loc[int(df.shape[0]*test_size):].reset_index().drop('index', axis=1)
    return train_df, test_df

def visualize_spec(df: pd.DataFrame, idx: int):
    '''
    Функция принимает на вход датасет и индекс и визуализирует соответствующую спектрограмму
    Входные данные:
    df: pd.DataFrame - датафрейм с данными
    idx: int - индекс визуализируемого элемента
    '''
    im = np.load(df.iloc[idx, 0])
    plt.imshow(im, cmap='jet')
    plt.show()
    print(df.iloc[idx, 1])

def get_mean_std(research_data_loader: DataLoader):
    '''
    Функция принимает на вход исследуемый датасет и вычисляет по нему среднее и СКО
    Входные данные:
    research_data_loader: DataLoader - датасет, приведенный к формату DataLoader
    '''
    nimages = 0
    mean = 0.
    std = 0.
    for batch in research_data_loader:
        nimages += 1
        mean += batch[0].mean()
        std += batch[0].std()

    mean /= nimages
    std /= nimages
    return float(mean), float(std)