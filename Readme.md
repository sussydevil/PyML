-----

Dataset — новости RT: 
https://github.com/ods-ai-ml4sg/proj_news_viz/releases/download/data/rt.csv.gz 

Необходимо на основании датасета разработать модель классификации новостей по категориям (Мир, Бывший СССР, России и тд). Модель можно реализовать с использованием любой библиотеки/фреймворка. Над моделью реализовать API с двумя методами:
1.  classify_text — в теле запроса передается текст новости, в ответе категории и процент соответствия
2.  classify_url — в параметре запроса url передается адрес страницы новости с https://russian.rt.com,  в ответе категории и процент соответствия основной статье страницы

Результат выложить открытым репозиторием на github.com, в репозиторий добавить docker-compose для запуска приложения с API.  API можно разработать с применением любого удобного фреймворка или на голом Python.

-----
Установка:

1) Скачивание с репозитория: `git clone https://github.com/sussydevil/PyML.git`.
Должен быть установлен Git LFS для скачивания большого *.csv файла. Проверка Git LFS - `git lfs install`, если его нет, нужно установить.
2) Вход в папку: `cd PyML`
3) Сборка образа: `docker build --tag aianalyzer .`
4) Запуск командой: `docker run -p 4000:4000 aianalyzer`

Пример клиентских запросов 2 методов находится в файле **jrpc_client.py**

P.S.
docker-compose не умеет сохранять state (https://stackoverflow.com/questions/44480740/how-to-save-a-docker-container-state), 
поэтому был использован Docker.

Скрипт в Docker настроен на train&run, сначала обучение, затем поднятие API. В последующие разы запускается сразу API.
Обучение может занять достаточное время (до часа).

-----

