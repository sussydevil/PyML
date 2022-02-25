FROM python:3.9-rc-slim-buster
WORKDIR /ai_app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3" , "main.py", "-mode", "train&run"]