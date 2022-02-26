# установка python
FROM python:3.9-rc-slim-buster

# рабочая директория
WORKDIR /analyzer-dir

# копирование списка библиотек
COPY requirements.txt /analyzer-dir

# установка библиотек
RUN pip3 install --upgrade pip --no-cache-dir -r requirements.txt

# установка chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable

# установка chromedriver
RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

# установка порта дисплея
ENV DISPLAY=:99

# копирование содержимого
COPY . /analyzer-dir

# выполнение скрипта
CMD [ "python3" , "main.py", "-mode", "train&run"]
