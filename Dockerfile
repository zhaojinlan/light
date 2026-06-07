FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# 优先使用阿里云镜像（国内加速），只用预编译 wheel 避免源码编译
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 \
    --only-binary :all: \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    -r requirements.txt
# Pre-bundle tiktoken cache at the correct location tiktoken checks
# On Linux, tiktoken defaults to /tmp/data-gym-cache/
RUN mkdir -p /tmp/data-gym-cache
COPY .cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4 /tmp/data-gym-cache/
COPY .cache/fb374d419588a4632f3f557e76b4b70aebbca790 /tmp/data-gym-cache/

COPY . .

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
