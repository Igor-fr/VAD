import pyaudio
import torchaudio
import torch
from model import ModelVad
import json

if __name__ == '__main__':
    #Размер пакета данных записи в отсчетах
    CHUNK = 5120
    #Формат данных
    FORMAT = pyaudio.paFloat32
    #Количество каналов
    CHANNELS = 1
    #Частота записи
    RATE = 16000
    #Окно
    WIN = int(RATE/100)

    with open('config.json', 'r') as f:
        config = json.load(f)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    #Значения среднего и СКО, вычисленные для спектрограмм на тестовом датасете
    mean = -3.052366018295288
    std = 2.4621522426605225

    model = ModelVad()
    model.load_state_dict(torch.load(config['path_to_checkpoint']))

    transformer = torchaudio.transforms.MelSpectrogram(sample_rate=RATE, win_length=WIN, n_mels=32, power=1, hop_length=WIN, n_fft=WIN+1)

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow = False)
        except:
            break
        data = torch.frombuffer(data, dtype=torch.float32).type(torch.FloatTensor)
        specgram = transformer(data)
        specgram = specgram.log2().detach().numpy()
        specgram = (specgram - mean) / std
        specgram = torch.FloatTensor(specgram.reshape(1, 1, 32, 32))
        out = model(specgram)
        print(float(out[0][0]))
        print('*'*int(float(out[0][0])*100))

    stream.stop_stream()
    stream.close()
    p.terminate()