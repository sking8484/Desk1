FROM python:3.10

WORKDIR /code

copy ./environment.txt ./
RUN pip install -r environment.txt

COPY ./src ./src
CMD ["python", "src/ella.py"]
