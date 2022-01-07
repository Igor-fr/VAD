### В проекте реализован VAD по материалам статьи [_\[source\]_](https://arxiv.org/pdf/2103.03529.pdf)
- training - папка, содержит файлы:
	- youtube.py - содержит методы для загрузки видео с youtube для подготовки обучающего AVA датасета 
	- prepare.py - содержит методы для обработки загруженных аудио, приведению их к спектрограммам и тд
	- model_wrapper.py - класс для работы с моделью
	- train_metrics.py - содержит в себе методы для оценки качества модели при обучении
- data - папка, содержит последний checkpoint модели
- demo.ipynb - блокнот, в котором выполняется подготовка данных, обучение модели и оценка итоговых метрик
- realtime.py - скрипт, реализующий работу итоговой модели в режиме реального времени на аудио, получаемом с микрофона
- metrics.py - скрипт получения метрик для итоговой модели на датасете в AVA-формате (по умолчанию принимается что датасет с неполными лейблами, то есть все фрагменты, не размеченные как SPEECH, размечаются как NO_SPEECH)
- config.json - содержит информацию для работы скриптов realtime и metrics, такую как путь к модели, путь к датасету и тд
- model.py - класс модели
- dataset.py - класс датасета для обучения