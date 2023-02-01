FROM amazon/aws-lambda-python:3.9
ARG function

COPY ./environment.txt ./
RUN pip install -r environment.txt

COPY ./src/$function ./
COPY ./src/datahub ./
CMD [ "src/ella.handler" ]
