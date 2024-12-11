FROM python:3.12.7

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./app /app/app

CMD ["streamlit", "run", "app/server.py", "--server.port=8080", "--server.address=0.0.0.0"]

##DOCKER IMAGE - [ docker container run -d -p 8080:8080 shubhamklizo/android_grade_streamlit:latest ]
## DOCKER CONTAINER - [ docker container run --name android_grade_streamlit  -d -p 8080:8080 shubhamklizo/android_grade_streamlit:latest]