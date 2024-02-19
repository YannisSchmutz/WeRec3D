FROM tensorflow/tensorflow:latest-gpu-jupyter

ADD requirements/ /requirements/
RUN set -ex \
  && pip3 install --upgrade pip \
  && pip3 install -r /requirements/requirements_server.txt

RUN mkdir /workspace
WORKDIR /workspace

# Also to keep it running:)
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
