# -*- coding: utf-8 -*-
from selenium.webdriver.support.wait import WebDriverWait
from keras.preprocessing.sequence import pad_sequences
from jsonrpc import JSONRPCResponseManager, dispatcher
from werkzeug.wrappers import Request, Response
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from selenium.webdriver.common.by import By
from werkzeug.serving import run_simple
from fake_useragent import UserAgent
from collections import OrderedDict
from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils
from selenium import webdriver
from termcolor import colored
from bs4 import BeautifulSoup
from keras import utils
from time import sleep

import selenium.common.exceptions
import keras.preprocessing.text
import pandas as pd
import numpy as np
import pymorphy2
import requests
import argparse
import pickle
import re
import os

# максимальное количество слов в словаре токенизатора
num_words = 500

# максимальная длина новости
max_news_len = 150

# количество классов новостей, считается ниже
nb_classes = 10

# объект для чистки текста
ma = pymorphy2.MorphAnalyzer()

# процент обучающей выборки
learn_percentage = 0.95

# путь сохранения модели
model_lstm_save_path = 'lstm_model.h5'

# путь исходного файла
original_csv_path = 'rt.csv'

# путь файла с категориями
categories_path = "categories.pkl"

# путь токенизатора
tokenizer_path = "tokenizer.json"

# глобальные переменные
global gl_model, gl_tokenizer, gl_categories

# настройки selenium
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1920,1080')
chrome_options.add_argument('--headless')
chrome_options.add_experimental_option("excludeSwitches", ['enable-automation'])
chrome_options.add_argument(f'user-agent={UserAgent().chrome}')
chrome_options.add_argument('--disable-gpu')
driver = webdriver.Chrome(options=chrome_options)


def file_cleaner(pandasf, column_array):
    """
    Очистка ненужных колонок и пустых строк
    Вход: исходный dataset, массив названий колонок
    Выход: очищенный dataset
    """
    for i in column_array:
        pandasf.drop(i, axis='columns', inplace=True)

    pandasf['text'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['text'], inplace=True)

    pandasf['topics'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['topics'], inplace=True)

    pandasf['title'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['title'], inplace=True)

    return pandasf


def text_cleaner(text):
    """
    Очистка текста от ненужных символов и предлогов
    Вход: грязный текст
    Выход: очищенный текст
    """

    # строчные буквы
    text = text.lower()

    # удаление лишних символов
    text = re.sub(r'\s\r\n\s+|\s\r\n|\r\n', '', text)
    text = re.sub(r'[.,:;_% ​"»«©?*/!@#$^&()\d]|[+=]|[\[]]|-|[a-z]|[A-Z]', ' ', text)

    # удаление предлогов
    text = ' '.join(word for word in text.split() if len(word) > 3)

    return text


def train():
    """
    Функция обучения
    """

    # чтение файла, удаление ненужной информации (даты, репосты и т.д.), очистка null строк
    print("-----------------------------------------------------------------")
    print(colored("Train module started...", "yellow"))
    print("Parameters: "
          "\n - Number of words: {0}"
          "\n - Max length of news: {1}"
          "\n - Learn percentage (volume): {2}".format(num_words, max_news_len, learn_percentage))
    print("-----------------------------------------------------------------")
    sleep(1)
    print(colored(">>> Reading {0}...", "yellow").format(original_csv_path))
    pandasf = pd.read_csv(original_csv_path)
    print(colored(">>> Reading done.", "green"))
    print("-----------------------------------------------------------------")

    # очистка текста от символов и предлогов
    print(colored(">>> Cleaning file of symbols, pretexts and unusable columns...", "yellow"))
    pandasf = file_cleaner(pandasf,
                           ['authors', 'date', 'url', 'edition', 'reposts_fb', 'reposts_vk', 'reposts_ok',
                            'reposts_twi', 'reposts_lj', 'reposts_tg', 'likes', 'views', 'comm_count'])
    pandasf['text'] = pandasf.apply(lambda x: text_cleaner(x['text']), axis=1)
    pandasf['title'] = pandasf.apply(lambda x: text_cleaner(x['title']), axis=1)
    print(colored(">>> Cleaning file of symbols, pretexts and unusable columns done.", "green"))
    print("-----------------------------------------------------------------")

    # выделение категорий
    categories = OrderedDict()
    print("Categories are (saved to {0}):".format(categories_path))
    for key, value in enumerate(pandasf['topics'].unique()):
        categories[value] = key
        print("{0}) {1}".format(key, value))

    # сохранение категорий в pickle
    with open(categories_path, 'wb') as f:
        pickle.dump(categories, f)

    # расчет количества категорий
    global nb_classes
    nb_classes = len(categories)
    print("Category count is: {0}".format(nb_classes))
    print("-----------------------------------------------------------------")

    # конвертирование категорий в числа
    pandasf['topics_code'] = pandasf['topics'].map(categories)

    # удаление topics из файла, т.к. есть topics_code
    pandasf.drop("topics", axis='columns', inplace=True)

    # перемешивание dataset
    pandasf = pandasf.sample(frac=1).reset_index(drop=True)

    # разбивание dataset, обычно для обучения используется 90-95%
    str_count = len(pandasf.index)
    pointer = int(str_count * learn_percentage)
    pandasf = pandasf.iloc[:pointer, :]

    # вычисление максимальной длины текста в словах
    max_words = 0
    for i in pandasf['text']:
        words = len(i.split())
        if words > max_words:
            max_words = words
    print('Max word count in news: {} words'.format(max_words))
    print("-----------------------------------------------------------------")

    # подготовка данных
    print(colored(">>> Dataset preparation...", "yellow"))
    news_text = pandasf['title'] + pandasf['text']
    y_train = utils.np_utils.to_categorical(pandasf['topics_code'], nb_classes)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(news_text)
    sequences = tokenizer.texts_to_sequences(news_text)
    x_train = pad_sequences(sequences, maxlen=max_news_len)
    print(colored(">>> Dataset preparation done.", "green"))
    print("-----------------------------------------------------------------")
    print(colored(">>> Learning started.", "yellow"))

    # LSTM нейронная сеть
    model_lstm = Sequential()
    model_lstm.add(Embedding(num_words, 32, input_length=max_news_len))
    model_lstm.add(LSTM(16))
    model_lstm.add(Dense(nb_classes, activation='softmax'))
    model_lstm.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    model_lstm.summary()

    # callback для сохранения нейронной сети на каждой эпохе
    checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path,
                                               monitor='val_accuracy',
                                               save_best_only=True,
                                               verbose=1)

    # обучение
    model_lstm.fit(x_train,
                   y_train,
                   epochs=5,
                   batch_size=128,
                   validation_split=0.1,
                   callbacks=[checkpoint_callback_lstm])

    # сохранение tokenizer в json
    with open(tokenizer_path, "w") as f:
        f.write(tokenizer.to_json())

    print(colored(">>> Learning done. Model saved to lstm_model.h5.", "green"))
    print("-----------------------------------------------------------------")


def get_key(d, value):
    """
    Функция получения ключа по значению
    Вход: словарь, значение
    Выход: ключ
    """

    for k, v in d.items():
        if v == value:
            return k


def api(text):
    """
    Функция для определения категории
    Вход: текст
    Выход: словарь категорий с вероятностями
    """

    # очистка текста новости
    text = text_cleaner(text)

    # обработка текста новости
    pandas_list = [text]
    pandasframe = pd.DataFrame(pandas_list, columns=['text'])
    text_sequences = gl_tokenizer.texts_to_sequences(pandasframe["text"])
    x = pad_sequences(text_sequences, maxlen=max_news_len)

    # распознавание категории нейросетью
    prediction = gl_model.predict(x).tolist()

    # выходной словарь
    output = OrderedDict()

    # заполнение словаря
    for i in range(len(prediction[0])):
        output[list(gl_categories.items())[i][0]] = prediction[0][i]

    # сортировка по значению
    output = sorted(output.items(), key=lambda y: y[1], reverse=True)

    return "{}".format(output)


def parse_site(url):
    """
    Функция парсинга url
    Вход: url
    Выход: текст новости -> словарь категорий с вероятностями
    """

    # получение html сайта
    response = requests.get(url, headers={'User-Agent': UserAgent().chrome})
    soup = BeautifulSoup(response.text, 'lxml')

    error = False

    # проверка на ошибку 403
    if response.status_code == 403:
        print(colored(">>> Access denied, error 403, trying selenium...", "red"))
        error = True

    # проверка на DDOS-GUARD
    elif response.status_code == 200:
        if soup.select_one('title').text == "DDOS-GUARD":
            print(colored(">>> Access denied, DDOS-Guard blocked access (title=DDOS-GUARD), trying selenium...", "red"))
            error = True

    # проверка на другие ошибки
    elif response.status_code not in {200, 403}:
        print(colored("Error {}, trying selenium...".format(response.status_code), "red"))
        error = True

    # если ошибка, то используется selenium
    if error:
        driver.get(url)

        # парсинг
        driver_data = driver.page_source
        soup = BeautifulSoup(driver_data, 'lxml')

        # проверка на DDOS-GUARD
        if soup.select_one('title').text == "DDOS-GUARD":
            print(colored(">>> Wait for redirecting to rt.com page...", "yellow"))
            wait = WebDriverWait(driver, 5)

            # ожидание переадресации
            try:
                wait.until(lambda driver2: driver2.find_element(by=By.TAG_NAME, value="title").text == "DDOS-GUARD")

            except selenium.common.exceptions.TimeoutException:
                print(colored(">>> Failed to parse data from rt.com by bs4, selenium.", "red"))
                return {"error": "Failed to parse by bs4, selenium."}

            # еще раз парсинг
            soup = BeautifulSoup(driver.page_source, 'lxml')

        # если элементы пустые, то ошибка
        if soup.select_one(".article__heading_article-page") is None and \
                soup.select_one('.article__summary_article-page') is None and \
                soup.select_one('.article__text_article-page p') is None:
            print(colored(">>> Failed to parse data from rt.com by bs4, selenium.", "red"))
            return {"error": "Failed to parse by bs4, selenium."}

        else:
            print(colored(">>> Selenium received data successfully. Title is {0}"
                          .format(soup.select_one(".article__heading_article-page")), "green"))

    # парсинг нужных блоков
    article = soup.select('.article__heading_article-page')
    article_summary = soup.select('.article__summary_article-page')
    article_text = soup.select('.article__text_article-page p')

    # переменные с готовым текстом
    parsed_article = ""
    parsed_summary = ""
    parsed_text = ""

    # запись текста в переменные
    for i in article:
        parsed_article = parsed_article + " " + i.text

    for i in article_summary:
        parsed_summary = parsed_summary + " " + i.text

    for i in article_text:
        parsed_text = parsed_text + " " + i.text

    all_text = parsed_article + " " + parsed_summary + " " + parsed_text
    api_answer = api(all_text)

    return api_answer


@Request.application
def application(request):
    """
    Функция JSON-RPC сервера
    """

    dispatcher["text"] = lambda text: api(text)
    dispatcher["link"] = lambda link: parse_site(link)
    response = JSONRPCResponseManager.handle(request.data, dispatcher)

    return Response(response.json, mimetype='application/json')


def main():
    """
    Главная функция - парсинг параметров и меню
    """

    print("AI news analyzer started (v.0.9.0)...")
    sleep(1)
    global gl_model, gl_categories, gl_tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', action='store', dest='mode', help='Mode can be "train", "run" or "train&run"')

    if parser.parse_args().mode == "train":
        train()
        exit(0)

    elif parser.parse_args().mode == "run":
        print(colored(">>> Loading model for recognizing...", "yellow"))

        # загрузка модели
        gl_model = load_model(model_lstm_save_path, compile=True)

        # загрузка tokenizer
        with open(tokenizer_path, "r") as f:
            tokenizer_string = f.read()
        gl_tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_string)

        # загрузка категорий
        with open(categories_path, 'rb') as f:
            gl_categories = pickle.load(f)
        print(colored(">>> Model loaded successfully...", "green"))

        run_simple('0.0.0.0', 4000, application)
        exit(0)

    elif parser.parse_args().mode == "train&run":
        if not os.path.exists(model_lstm_save_path):
            train()
        else:
            print(colored('>>> Model found. Skipping training. '
                          'If you to train a new model, reload app with "-mode train"', "green"))

        print(colored(">>> Loading model for recognizing...", "yellow"))

        # загрузка модели
        gl_model = load_model(model_lstm_save_path, compile=True)

        # загрузка tokenizer
        with open(tokenizer_path, "r") as f:
            tokenizer_string = f.read()
        gl_tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_string)

        # загрузка категорий
        with open(categories_path, 'rb') as f:
            gl_categories = pickle.load(f)
        print(colored(">>> Model loaded successfully...", "green"))

        run_simple('0.0.0.0', 4000, application)
        exit(0)

    else:
        print(colored("Argv not recognized. Then menu is on.", "red"))

    error = False

    while True:
        if not error:
            print(colored("Please choose what you want to do:\n1) Train model\n2) Run model (enable API)", "yellow"))
        answer = input('>>> ')
        try:
            answer = int(answer)
        except ValueError:
            print(colored('Convert error to int. Try again.', 'red'))
            error = True
            continue
        if answer not in {1, 2}:
            print(colored("Selection error. Try again.", "red"))
            error = True
            continue
        if answer == 1:
            train()
        if answer == 2:
            print(colored(">>> Loading model for recognizing...", "yellow"))

            # загрузка модели
            gl_model = load_model(model_lstm_save_path, compile=True)

            # загрузка tokenizer
            with open(tokenizer_path, "r") as f:
                tokenizer_string = f.read()
            gl_tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_string)

            # загрузка категорий
            with open(categories_path, 'rb') as f:
                gl_categories = pickle.load(f)
            print(colored(">>> Model loaded...", "green"))

            run_simple('0.0.0.0', 4000, application)

        error = False


if __name__ == '__main__':
    main()
