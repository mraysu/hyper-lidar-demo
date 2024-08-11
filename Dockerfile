FROM ghcr.io/selkies-project/nvidia-glx-desktop:latest

COPY . semantic-kitti-api
COPY requirements.txt requirements.txt

#SHELL ["/bin/bash", "-c"]

RUN sudo apt-get update
RUN sudo apt install python3.12-venv -y
#RUN sudo python3 -m venv .venv
#RUN source .venv/bin/activate
#RUN .venv/bin/pip install -r requirements.txt
