# based off latest tensorflow
FROM tensorflow/tensorflow:latest-devel-gpu-py3

# add python3.6
RUN add-apt-repository ppa:jonathonf/python-3.6 -y
RUN apt-get update
RUN apt-get install python3.6 -y
RUN apt-get install python3.6-dev -y
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

# copy all data and source
COPY . /xray/

# set dir ready
WORKDIR /xray

# requirements
RUN pip3.6 install -r requirements.txt

# for open cv
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libgtk2.0-dev
