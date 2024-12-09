FROM dashinfer/dev-ubuntu-22.04-x86:v1_py310

WORKDIR /workspace

COPY ./allspark_worker.py ./
COPY ./fschat_entrypoint.sh ./

SHELL ["conda", "run", "-n", "test_py", "/bin/bash", "-c"]
RUN pip install \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  "fschat[model_worker]==0.2.36"

# fastchat has a bug in pydantic v2: https://github.com/lm-sys/FastChat/pull/3356
# downgrade to v1.10.13
RUN pip uninstall pydantic -y \
    && pip install -i https://mirrors.aliyun.com/pypi/simple pydantic==1.10.13

RUN pip install 'dashinfer==2.0.0'

RUN chmod +x ./fschat_entrypoint.sh

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "test_py", "./fschat_entrypoint.sh"]


