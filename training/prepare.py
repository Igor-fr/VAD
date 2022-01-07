import os
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
WIN_LENGTH = HOP_LENGTH = N_FFT = int(SAMPLE_RATE / 100)
STEP = int(0.32 * SAMPLE_RATE)
TRANSFORMER = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, win_length=WIN_LENGTH,
                                                                n_mels=32, power=1, hop_length=HOP_LENGTH,
                                                                n_fft=N_FFT + 1)

def read_speech_labels(speech_labels_paths: str):
    '''
    Функция принимает на вход путь к датасету со всеми аудио и возвращает его в виде массива
    Входные данные:
    speech_labels_paths: str - путь к датасету
    '''
    with open(speech_labels_paths) as f:
        speech_labels = f.readlines()
    return [x.replace('\n', '') for x in speech_labels]

def create_dir(path_to_dir: str):
    '''
    Функция принимает на вход путь к папке и создает ее в случае отсутствия
    Входные данные:
    path_to_dir: str - путь к папке
    '''
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

def create_spec(waveform: object, time_start: int):
    '''
    Функция принимает на вход аудио в виде тензора, вырезает из него фрагмент заданной длины с заданного момента
    времени и возвращает спектрограмму этого фрагмента
    Входные данные:
    waveform: object - аудио
    time_start: int - момент начала фрагмента в отсчетах
    '''
    specgram = TRANSFORMER(waveform[0][time_start:time_start + STEP])
    data = specgram.log2().detach().numpy()
    if True in np.isinf(data):
        return None
    return data

def prepare_speech_fragment(waveform: object, start_time: int, end_time: int, label: str, path_specs: str, audio_name: str):
    '''
    Функция принимает на вход аудио в виде тензора, и заданный фрагмент разбивает на фрагменты длительностью 0.32
    вычисляет для них спектрограммы, сохраняет их и возвращает список путей к спектрограммам с их метками класса
    Входные данные:
    waveform: object - аудио
    start_time: int - момент начала фрагмента в секундах
    end_time: int - момент окончания фрагмента в секундах
    label: str - метка класса для фрагмента
    path_specs: str - путь к папке для сохранения спектрограмм
    audio_name: str - имя аудио
    '''
    analize_second = start_time
    labels = []
    while (analize_second * SAMPLE_RATE < waveform.shape[1] - STEP) and (end_time - analize_second > 0.32):
        data = create_spec(waveform, int(analize_second * SAMPLE_RATE))
        if data is None:
            analize_second += 0.32
            continue
        spec_name = str(round(analize_second,3)).replace('.', '_')
        np.save(path_specs + '/{0}_{1}'.format(audio_name, spec_name), data)
        labels.append([path_specs + '/{0}_{1}.npy'.format(audio_name, spec_name), label])
        analize_second += 0.32
    return labels

def create_spec_dataset(speech_labels_paths: str, path_specs: str, delimiter: str = ','):
    '''
    Функция принимает на вход пути ко всем аудио датасета и преобразует их к спектрограммам
    Входные данные:
    speech_labels_paths: str - путь к датасету AVA формата
    path_specs: str - путь к папке, куда сохранять спектрограммы
    delimeter: str=',' - разделитель в датасете
    '''

    speech_labels = read_speech_labels(speech_labels_paths)
    create_dir(path_specs)

    i = 0
    labels = []

    while i < len(speech_labels):
        path_to_audio = speech_labels[i].split(delimiter)[0]
        audio = path_to_audio.split('/')[-1].split('.')[0]
        try:
            waveform, _ = torchaudio.load(path_to_audio)
            print(path_to_audio)
        except:
            i += 1
            continue
        while True:
            speech_label_split = speech_labels[i].split(delimiter)
            second_start = float(speech_label_split[1])
            second_end = float(speech_label_split[2])
            label = speech_label_split[3]
            if label == 'NO_SPEECH':
                label = '0'
            else:
                label = '1'
            labels += prepare_speech_fragment(waveform, second_start, second_end, label, path_specs, audio)
            i += 1
            if i == len(speech_labels):
                break
            if speech_labels[i].split(delimiter)[0] != path_to_audio:
                break
    return np.array(labels)

def create_spec_dataset_only_speech(speech_labels_paths: str, path_specs: str, delimiter: str = ','):
    '''
    Функция принимает на вход пути ко всем аудио датасета и преобразует их к спектрограммам, с учетом неполных лейблов,
    то есть в датасете есть только метки класса SPEECH, и все остальные фрагменты автоматически принимаются как
    NO_SPEECH
    Входные данные:
    speech_labels_paths: str - путь к датасету AVA формата
    path_specs: str - путь к папке, куда сохранять спектрограммы
    delimeter: str=',' - разделитель в датасете
    '''
    speech_labels = read_speech_labels(speech_labels_paths)
    create_dir(path_specs)

    i = 0
    labels = []

    while i < len(speech_labels):
        path_to_audio = speech_labels[i].split(delimiter)[0]
        audio = path_to_audio.split('/')[-1].split('.')[0]
        try:
            waveform, _ = torchaudio.load(path_to_audio)
            print(path_to_audio)
        except:
            i += 1
            continue
        analize_second = 0
        while True:
            speech_label_split = speech_labels[i].split(delimiter)
            second_start = float(speech_label_split[1])
            second_end = float(speech_label_split[2])
            if second_start > analize_second:
                labels += prepare_speech_fragment(waveform, analize_second, second_start, '0', path_specs, audio)
            analize_second = second_start
            labels += prepare_speech_fragment(waveform, analize_second, second_end, '1', path_specs, audio)
            analize_second = second_end
            i += 1
            if i == len(speech_labels):
                if analize_second * SAMPLE_RATE < waveform.shape[1] - STEP:
                    labels += prepare_speech_fragment(waveform, analize_second, waveform.shape[1], '0', path_specs, audio)
                break
            if speech_labels[i].split(delimiter)[0] != path_to_audio:
                if analize_second * SAMPLE_RATE < waveform.shape[1] - STEP:
                    labels += prepare_speech_fragment(waveform, analize_second, waveform.shape[1], '0', path_specs, audio)
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