from keras.preprocessing.text import Tokenizer
from numpy import compat
from time import sleep
import pandas as pd
import numpy as np
import pymorphy2
import os
import re


ma = pymorphy2.MorphAnalyzer()


def file_cleaner(pandasf, column_array):
    """
    Очистка ненужных колонок и пустых строк
    Вход: исходный файл, массив названий колонок
    """
    print("Deleting unused columns...")
    for i in column_array:
        pandasf.drop(i, axis='columns', inplace=True)
    print("Done.")

    print("Deleting null rows...")
    pandasf['text'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['text'], inplace=True)

    pandasf['topics'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['topics'], inplace=True)

    pandasf['title'].replace('', np.nan, inplace=True)
    pandasf.dropna(subset=['title'], inplace=True)
    print("Done.")

    return pandasf


def text_cleaner(text):
    """
    Очистка текста от ненужных символов и предлогов
    Вход: грязный текст
    Выход: очищенный текст
    """
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub(r'\s\r\n\s+|\s\r\n|\r\n', '', text)
    text = re.sub(r'[.,:;_%©?*!@#$^&()\d]|[+=]|[\[]]|[/]|"|\s{2,}|-', ' ', text)
    text = " ".join(ma.parse(compat.unicode(word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) > 3)
    text = text.encode("utf-8")
    return text


def load_data_from_arrays(strings, labels, train_test_split=0.9):
    """
    Разбивание Dataset на обучающую и проверочную выборки, по умолчанию 90% - обучающая
    Вход: Dataset, % обучающей выборки
    Выход: выборки
    """
    data_size = len(strings)
    test_size = int(data_size - round(data_size * train_test_split))
    print("Test size: {}".format(test_size))

    print("\nTraining set:")
    x_train = strings[test_size:]
    print("\t - x_train: {}".format(len(x_train)))
    y_train = labels[test_size:]
    print("\t - y_train: {}".format(len(y_train)))

    print("\nTesting set:")
    x_test = strings[:test_size]
    print("\t - x_test: {}".format(len(x_test)))
    y_test = labels[:test_size]
    print("\t - y_test: {}".format(len(y_test)))

    return x_train, y_train, x_test, y_test


def main():
    """
    Главная функция
    """
    # чтение файла, удаление ненужной информации (даты, репосты и т.д.)
    if not os.path.exists('rt.pkl'):
        print("File without unusable columns and null rows not found, creating from original...")
        pandasf = pd.read_csv('rt.csv')
        pandasf = file_cleaner(pandasf,
                               ['authors', 'date', 'url', 'edition', 'reposts_fb', 'reposts_vk', 'reposts_ok',
                                'reposts_twi', 'reposts_lj', 'reposts_tg', 'likes', 'views', 'comm_count'])
        pandasf.to_pickle('rt.pkl')
        print("File created successfully.")
    else:
        print("File without unusable columns and null rows found.")
        pandasf = pd.read_pickle('rt.pkl')

    # очистка текста
    if not os.path.exists("rt_clean.pkl"):
        print("Clean file not found, creating... It may take a long time.")
        pandasf['clean_text'] = pandasf.apply(lambda x: text_cleaner(x['text']), axis=1)
        pandasf.to_pickle("rt_clean.pkl")
        print("Cleaning done.")
    else:
        print("Clean file found.")
        pandasf = pd.read_pickle("rt_clean.pkl")

    # выделение категорий
    categories = {}
    for key, value in enumerate(pandasf['topics'].unique()):
        categories[value] = key
    pandasf['topics_code'] = pandasf['topics'].map(categories)
    total_categories = len(pandasf['topics'].unique())

    print('Всего категорий: {0}, {1}'.format(total_categories, pandasf['topics'].unique()))

    # перемешивание Dataset
    pandasf = pandasf.sample(frac=1).reset_index(drop=True)

    text = pandasf['clean_text']
    topics = pandasf['topics_code']

    # максимальная длина текста в словах
    max_words = 0
    for i in text:
        words = len(i.split())
        if words > max_words:
            max_words = words
    print('Max word count in texts: {} words'.format(max_words))

    # единый словарь (слово -> число) для преобразования
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([x.decode('utf-8') for x in text.tolist()])

    # преобразование всех описаний в числовые последовательности, заменяя слова на числа по словарю
    text_sequences = tokenizer.texts_to_sequences([x.decode('utf-8') for x in text.tolist()])

    # разбивание Dataset на обучающую и проверочную выборки
    x_train, y_train, x_test, y_test = load_data_from_arrays(text_sequences, topics, train_test_split=0.8)

    # вывод количества всех слов
    total_words = len(tokenizer.word_index)
    print('{} words in dictionary'.format(total_words))


if __name__ == '__main__':
    main()
