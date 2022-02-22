# -*- coding: utf-8 -*-
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras import utils
from time import sleep
import pandas as pd
import numpy as np
import pymorphy2
import re


# максимальное количество слов
num_words = 10000
# максимальная длина новости
max_news_len = 2000
# количество классов новостей
nb_classes = 10
# объект для чистки текста
ma = pymorphy2.MorphAnalyzer()


def file_cleaner(pandasf, column_array):
    """
    Очистка ненужных колонок и пустых строк
    Вход: исходный файл, массив названий колонок
    """
    print("Deleting unused columns and null rows...")
    for i in column_array:
        pandasf.drop(i, axis='columns', inplace=True)

    pandasf['text'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['text'], inplace=True)

    pandasf['topics'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['topics'], inplace=True)

    pandasf['title'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['title'], inplace=True)
    print("Deleting unused columns and null rows done.")

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


def main():
    """
    Главная функция
    """
    print("News analyzer started (v.0.1.0)...")
    sleep(1)
    # чтение файла, удаление ненужной информации (даты, репосты и т.д.), очистка null строк
    pandasf = pd.read_csv('rt.csv')
    pandasf = file_cleaner(pandasf,
                           ['authors', 'date', 'url', 'edition', 'reposts_fb', 'reposts_vk', 'reposts_ok',
                            'reposts_twi', 'reposts_lj', 'reposts_tg', 'likes', 'views', 'comm_count'])

    # очистка текста от символов и предлогов
    print("Cleaning of symbols and pretexts...")
    pandasf['text'] = pandasf.apply(lambda x: text_cleaner(x['text']), axis=1)
    pandasf['title'] = pandasf.apply(lambda x: text_cleaner(x['title']), axis=1)
    print("Cleaning of symbols and pretexts done.")

    # выделение категорий
    categories = {}
    for key, value in enumerate(pandasf['topics'].unique()):
        categories[value] = key
    print("Categories: {0}".format(categories))

    # конвертирование категорий в числа
    pandasf['topics_code'] = pandasf['topics'].map(categories)

    # удаление topics из файла, т.к. есть topics_code
    pandasf.drop("topics", axis='columns', inplace=True)

    # перемешивание Dataset
    pandasf = pandasf.sample(frac=1).reset_index(drop=True)

    # вычисление максимальной длины текста в словах
    max_words = 0
    for i in pandasf['text']:
        words = len(i.split())
        if words > max_words:
            max_words = words
    print('Max word count in news: {} words'.format(max_words))

    # подготовка данных
    news_text = pandasf['text']
    y_train = utils.np_utils.to_categorical(pandasf['topics_code'], nb_classes)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(news_text)
    sequences = tokenizer.texts_to_sequences(news_text)
    x_train = pad_sequences(sequences, maxlen=max_news_len)

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
    model_lstm_save_path = 'lstm_model.h5'
    checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path,
                                               monitor='val_accuracy',
                                               save_best_only=True,
                                               verbose=1)

    history_lstm = model_lstm.fit(x_train,
                                  y_train,
                                  epochs=5,
                                  batch_size=128,
                                  validation_split=0.1,
                                  callbacks=[checkpoint_callback_lstm])

    # построение графика в matplotlib
    plt.plot(history_lstm.history['accuracy'], label='Доля верных ответов на обучающем наборе')
    plt.plot(history_lstm.history['val_accuracy'], label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()

    # загрузка набора данных для тестирования
    test = pd.read_csv('test.csv',
                       header=None,
                       names=['class', 'title', 'text'])

    # преобразование новости в числовое представление, используя токенизатор, обученный на наборе данных train
    test_sequences = tokenizer.texts_to_sequences(test['text'])
    x_test = pad_sequences(test_sequences, maxlen=max_news_len)

    # Правильные ответы
    y_test = utils.np_utils.to_categorical(test['class'] - 1, nb_classes)

    # оценка качества работы сети на тестовом наборе данных
    model_lstm.load_weights(model_lstm_save_path)
    model_lstm.evaluate(x_test, y_test, verbose=1)


if __name__ == '__main__':
    main()
