FROM amazon/aws-lambda-python:3.9
ARG function

COPY ./requirements.txt ./
RUN pip install -r requirements.txt


COPY ./src/$function ./src/main
COPY ./src/db_link ./src/db_link

CMD [ "src/main/handler.handler" ]
