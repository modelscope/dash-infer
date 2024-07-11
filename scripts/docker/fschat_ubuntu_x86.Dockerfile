FROM dashinfer/base_ubuntu_x86:v1.2.1

WORKDIR /workspace

COPY ./dashinfer_worker.py ./
COPY ./fschat_entrypoint.sh ./

SHELL ["conda", "run", "-n", "py38env", "/bin/bash", "-c"]
RUN pip install \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  "fschat[model_worker]"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "py38env", "./fschat_entrypoint.sh"]
