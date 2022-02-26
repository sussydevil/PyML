# установка python
FROM python:3.9-rc-slim-buster

# рабочая директория
WORKDIR /analyzer-dir

# копирование списка библиотек
COPY requirements.txt /analyzer-dir

# установка библиотек
RUN pip3 install --upgrade pip --no-cache-dir -r requirements.txt

# копирование содержимого
COPY . /analyzer-dir

# выполнение скрипта
CMD [ "python3" , "main.py", "-mode", "train&run"]
