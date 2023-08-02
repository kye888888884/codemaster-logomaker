FROM python:3.7-slim
WORKDIR /usr/src/app
COPY . .
CMD [ "test.py" ]
ENTRYPOINT [ "python3" ]