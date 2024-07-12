FROM dashinfer/base_ubuntu_x86:v1.2.1

WORKDIR /workspace

COPY ./dashinfer_worker.py ./
COPY ./fschat_entrypoint.sh ./

SHELL ["conda", "run", "-n", "py38env", "/bin/bash", "-c"]
RUN pip install \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  "fschat[model_worker]==0.2.36"

# fastchat has a bug in pydantic v2: https://github.com/lm-sys/FastChat/pull/3356
# downgrade to v1.10.13
RUN pip uninstall pydantic -y \
    && pip install -i https://mirrors.aliyun.com/pypi/simple pydantic==1.10.13

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "py38env", "./fschat_entrypoint.sh"]
