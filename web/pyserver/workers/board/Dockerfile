FROM python:3.7

RUN apt update && apt install -y libsasl2-dev python-dev libldap2-dev libssl-dev

COPY ./workers/board/requirements.txt /code/workers/board/requirements.txt
RUN pip install -r /code/workers/board/requirements.txt

ADD . /code
ENV PYTHONPATH="/code"

CMD ["python", "-u", "/code/workers/board/main.py"]
