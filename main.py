# -*- coding: utf-8 -*-
import keras.preprocessing.text
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils
from termcolor import colored
from keras import utils
from time import sleep

import pandas as pd
import numpy as np
import pymorphy2
import argparse
import pickle
import re

# максимальное количество слов
num_words = 100000
# максимальная длина новости
max_news_len = 2500
# количество классов новостей, считается ниже
nb_classes = 0
# объект для чистки текста
ma = pymorphy2.MorphAnalyzer()
# процент обучающей выборки
learn_percentage = 0.95
# путь сохранения модели
model_lstm_save_path = 'lstm_model.h5'


def file_cleaner(pandasf, column_array):
    """
    Очистка ненужных колонок и пустых строк
    Вход: исходный файл, массив названий колонок
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
    text = text.lower()
    text = re.sub(r'\s\r\n\s+|\s\r\n|\r\n', '', text)
    text = re.sub(r'[.,:;_% ​"»«©?*/!@#$^&()\d]|[+=]|[\[]]|-|[a-z]|[A-Z]', ' ', text)
    text = ' '.join(word for word in text.split() if len(word) > 3)

    return text


def train():
    """
    Функция обучения
    """
    # чтение файла, удаление ненужной информации (даты, репосты и т.д.), очистка null строк
    print("-----------------------------------------------------------------")
    print(colored("Train module started...\nParameters: "
                  "\nNumber of words: {0}"
                  "\nMax length of news: {1}"
                  "\nLearn percentage (volume): {2}", "yellow").format(num_words, max_news_len, learn_percentage))
    print("-----------------------------------------------------------------")
    sleep(2)
    print(colored(">>> Reading rt.csv...", "yellow"))
    pandasf = pd.read_csv('rt.csv')
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
    categories = {}
    print("Categories are (saved to categories.pkl):")
    for key, value in enumerate(pandasf['topics'].unique()):
        categories[value] = key
        print("{0}) {1}".format(key, value))

    # сохранение категорий в pickle
    with open('categories.pkl', 'wb') as f:
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

    # перемешивание Dataset
    pandasf = pandasf.sample(frac=1).reset_index(drop=True)

    # разбивание Dataset, обычно для обучения используется 90-95%
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
    news_text = pandasf['text']
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
    model_lstm.add(Dense(10, activation='softmax'))
    model_lstm.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    model_lstm.summary()

    # callback для сохранения нейронной сети на каждой эпохе
    checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path,
                                               monitor='val_accuracy',
                                               save_best_only=True,
                                               verbose=1)

    model_lstm.fit(x_train,
                   y_train,
                   epochs=5,
                   batch_size=128,
                   validation_split=0.1,
                   callbacks=[checkpoint_callback_lstm])

    # сохранение tokenizer в json
    with open("tokenizer.json", "w") as f:
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


def api():
    # загрузка модели
    model = load_model(model_lstm_save_path, compile=True)

    # загрузка tokenizer
    with open("tokenizer.json", "r") as f:
        tokenizer_string = f.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_string)

    with open('categories.pkl', 'rb') as f:
        categories = pickle.load(f)

    # вывод информации о модели
    model.summary()

    """
    # загрузка тестовых данных
    pandasf = pd.read_csv("clean.csv")
    pandasf = pandasf.iloc[100000:100100, :]
    pandasf['text'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['text'], inplace=True)
    text_array = pandasf['text']

    # преобразование тестовых данных
    test_sequences = tokenizer.texts_to_sequences(text_array)
    x_test = pad_sequences(test_sequences, maxlen=max_news_len)
    predictions = np.argmax(model.predict(x_test), axis=1)
    ans = pandasf['topics_code'].to_numpy()

    for i in range(len(ans)):
        pre = get_key(categories, predictions[i])
        nast = get_key(categories, ans[i])
        if int(predictions[i]) != int(ans[i]):
            print(colored("{0} -> {1} | {2} -> {3}", "red").format(predictions[i], ans[i], pre, nast))
        else:
            print(colored("{0} -> {1} | {2} -> {3}", "green").format(predictions[i], ans[i], pre, nast))
    """

    while True:
        text = input("Введите текст: ")
        text = text_cleaner(text)
        test_dict = [text]

        df = pd.DataFrame(test_dict, columns=['text'])

        test_sequ = tokenizer.texts_to_sequences(df["text"])
        x_test_ = pad_sequences(test_sequ, maxlen=max_news_len)
        prediction = np.argmax(model.predict(x_test_), axis=1)
        print("{0} - {1}".format(prediction, get_key(categories, prediction)))


def main():
    """
    Главная функция
    """
    print("AI news analyzer started (v.0.3.0)...")
    sleep(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', action='store', dest='mode', help='Mode can be "train" or "api"')

    if parser.parse_args().mode == "train":
        train()
        exit(0)
    elif parser.parse_args().mode == "api":
        api()
        exit(0)
    else:
        print(colored("Argv not recognized. Then menu is on.", "red"))

    error = False

    while True:
        if not error:
            print(colored("Please choose what you want to do:\n1) Train model\n2) Use model (enable API)", "yellow"))
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
            api()
        error = False


if __name__ == '__main__':
    main()
