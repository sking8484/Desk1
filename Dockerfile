FROM amazon/aws-lambda-python:3.9
ARG function

COPY ./requirements.txt ./
RUN pip install -r requirements.txt


COPY ./src/$function ./src/main
COPY ./src ./src
RUN python src/main/handler.py 
CMD [ "src/main/handler.handler" ]
