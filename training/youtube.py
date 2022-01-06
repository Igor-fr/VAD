from pytube import YouTube
import glob
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def download_audio_from_youtube(video_dataset_path: str, audio_save_dir_path: str):
    '''
    Функция загружает все аудио из спика видео c youtube в формет mp4
    Входные параметры:
    video_dataset_path: str - путь к файлу со списком видео
    audio_save_dir_path: str - путь к папке, куда сохранять аудио
    '''
    with open(video_dataset_path) as f:
        links = f.readlines()
    links = [x.replace('\n', '') for x in links]
    
    if not os.path.exists(audio_save_dir_path):
        os.makedirs(audio_save_dir_path)
        
    for link in links:
        yt=YouTube('https://www.youtube.com/watch?v=' + link)
        try:
            t=yt.streams.filter(only_audio=True)
            t[0].download(audio_save_dir_path)
            print(os.path.join(audio_save_dir_path, link + '.mp4'))
            os.rename(os.path.join(audio_save_dir_path, t[0].default_filename), 
                      os.path.join(audio_save_dir_path, link + '.mp4'))
        except:
            continue

def convert_audio_to_wav(audio_mp4_dir_path:str, audio_wav_dir_save_path:str):
    '''
    Функция конвертирует все аудио из спика из формата mp4 в формат mav с частотой 16кГц, 1 канал
    Входные параметры:
    audio_mp4_dir_path: str - путь к папке с аудио в mp4
    audio_wav_dir_save_path: str - путь к папке, куда сохранять аудио в формате wav
    '''
    if not os.path.exists(audio_wav_dir_save_path):
        os.makedirs(audio_wav_dir_save_path)
        
    for audio in glob.glob(audio_mp4_dir_path + "/*.mp4"):
        _, file = os.path.split(audio)
        command = "ffmpeg -i {} -ab 160k -ac 1 -ar 16000 -vn {}".format(audio,
                                                                           os.path.join(audio_wav_dir_save_path,
                                                                                        file.split('.')[0] + '.wav'))

        print(os.path.join(audio_wav_dir_save_path, file.split('.')[0] + '.wav'))
        subprocess.call(command, shell=True)

def dataset_to_abs_path(speech_labels_paths: str, path_to_audio_dir: str, delimiter:str = ','):
    '''
    Функция конвертирует датасет с относительными путями к датасету с абсолютными путями:
    Входные параметры:
    speech_labels_paths: str - путь к датасету с аудио
    path_to_audio_dir: str - путь к папке с аудио wav
    delimiter:str = ',' - разделитель в датасете
    '''
    with open(speech_labels_paths) as f:
        speech_labels = f.readlines()
        
    speech_labels = [path_to_audio_dir + '/' + x.split(delimiter)[0] + '.wav,' + delimiter.join(x.split(delimiter)[1:]) for x in speech_labels]
    with open(speech_labels_paths, 'w') as f:
        f.writelines(speech_labels)