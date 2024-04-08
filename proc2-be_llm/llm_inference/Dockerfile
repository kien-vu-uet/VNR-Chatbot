FROM python:3.8

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install torch==1.12.1+cu113 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu113 

EXPOSE 9104

CMD ["uvicorn", "run:app", "--host=0.0.0.0", "--port=5002"]