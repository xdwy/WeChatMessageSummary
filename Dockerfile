FROM python:3.10
COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -i https://pypi.org/simple

CMD ["python", "main.py"]

# 启动命令
# docker build -t messageprocessor .
# docker run -d --name wechatApp --restart always -p 9700:9700 -v $(pwd):/app --privileged messageprocessor:latest