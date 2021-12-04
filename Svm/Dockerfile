FROM python:3.7-alpine

RUN mkdir /app
WORKDIR /app  
ADD requirements.txt /app
ADD main.py /app

RUN python -m pip install --upgrade pip && pip3 install -r requirements.txt 


CMD ["gunicorn","--workers","4", "--bind", "0.0.0.0:5000", "main:app"]