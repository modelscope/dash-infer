FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /root/

RUN mkdir /root/code/
COPY ./dashinfer_vlm /root/code/dashinfer_vlm
COPY ./setup.py code/
COPY ./requirements.txt /root/code/requirements.txt
RUN python3 -m pip install https://github.com/modelscope/dash-infer/releases/download/v2.0.0-rc2/dashinfer-2.0.0rc2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN python3 -m pip install -r /root/code/requirements.txt --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

RUN python3 -m pip install -e /root/code/

COPY ./entrypoint.sh /root/entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT  ["./entrypoint.sh"]
