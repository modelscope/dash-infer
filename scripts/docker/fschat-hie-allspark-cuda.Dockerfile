FROM reg.docker.alibaba-inc.com/hci/base-hie-allspark-cuda:3.0.2

WORKDIR /root/workspace


COPY ./fschat_entrypoint.sh ./
COPY ./allspark_worker.py ./

RUN chmod +x ./fschat_entrypoint.sh

SHELL [ "conda", "run", "--no-capture-output", "-n", "py38", "/bin/bash", "-c" ]
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple \
        addict \
        modelscope \
        psutil \
        accelerate \
        "fschat==0.2.36"

# fastchat has a bug in pydantic v2: https://github.com/lm-sys/FastChat/pull/3356
# downgrade to v1.10.13
RUN pip3 uninstall pydantic -y \
    && pip3 install -i https://mirrors.aliyun.com/pypi/simple pydantic==1.10.13

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "py38", "./fschat_entrypoint.sh"]