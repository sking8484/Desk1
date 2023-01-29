FROM amazon/aws-lambda-python:3.9

copy ./environment.txt ./
RUN pip install -r environment.txt

COPY ./src ./
CMD [ "src/ella.handler" ]
