FROM tensorflow/tensorflow:latest-gpu
LABEL AUTHOR abhijithganesh

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . . 
