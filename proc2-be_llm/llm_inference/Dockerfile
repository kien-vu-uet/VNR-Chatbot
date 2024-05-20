FROM python:3.8

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install --no-cache-dir -r requirements.txt
    # pip install torch==2.0.1

EXPOSE 9104

CMD ["uvicorn", "run:app", "--host=0.0.0.0", "--port=5002"]