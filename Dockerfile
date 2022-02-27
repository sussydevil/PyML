# установка ubuntu
FROM python:3.9-rc-slim-buster

# обновление пакетов
RUN apt-get update -y

# установка нужных программ
RUN apt-get install -y sudo
RUN apt-get install -y wget
RUN apt-get install -y unzip
RUN apt-get install -y curl

# рабочая директория
WORKDIR /analyzer-dir

# установка chrome
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y gnupg2
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get update -y
RUN apt-get install -y google-chrome-stable

# установка chromedriver
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

# установка порта дисплея
ENV DISPLAY=:99

# копирование списка библиотек
COPY requirements.txt /analyzer-dir

# установка библиотек
RUN pip3 install --upgrade pip --no-cache-dir -r requirements.txt

# копирование содержимого
COPY . /analyzer-dir

# выполнение скрипта
CMD [ "python3" , "main.py", "-mode", "train&run"]
