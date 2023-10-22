FROM python:3.9-slim as compile-image

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.9 as build-image
ARG function

COPY --from=compile-image /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY ./src/$function ./src/main
COPY ./src ./src
RUN python -W ignore:PendingDeprecationWarning -m unittest discover -s src -vvv -f
CMD [ "python","src/main/handler.py" ]
