FROM ubuntu:18.04

RUN mkdir /usr/src/app

WORKDIR /usr/src/app

ADD . /usr/src/app

RUN apt-get update 
RUN apt-get install -y && apt-get install -y python3.8 && apt-get install python3-pip -y

RUN pip3 install Flask==1.1.2
RUN pip3 install pandas
RUN pip3 install pillow==6.2.2
RUN pip3 install matplotlib
RUN apt-get install llvm-10 lldb-10 llvm-10-dev libllvm10 llvm-10-runtime -y
RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-10 10 
RUN update-alternatives --config llvm-config
RUN pip3 install librosa
RUN apt-get install libsndfile1 -y

RUN pip3 install keras==2.3.1
RUN pip3 install tensorflow 

EXPOSE 5000
CMD ["python3","main.py"]