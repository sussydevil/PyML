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
    pandasf['category_code'] = pandasf['topics'].map(categories)
    total_categories = len(pandasf['topics'].unique())

    print('Всего категорий: {0}, {1}'.format(total_categories, pandasf['topics'].unique()))


if __name__ == '__main__':
    main()
