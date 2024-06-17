FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

ARG function

COPY ./src/$function ${LAMBDA_TASK_ROOT}/src/main
COPY ./src ${LAMBDA_TASK_ROOT}
COPY ./src/__init__.py ${LAMBDA_TASK_ROOT}
COPY ./src/__init__.py ${LAMBDA_TASK_ROOT}/src
COPY ./src/__init__.py ${LAMBDA_TASK_ROOT}/src/main

CMD [ "src.main.handler.handler" ]
