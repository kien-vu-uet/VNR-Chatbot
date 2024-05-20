FROM scada-similarity-checking:latest

WORKDIR /app

COPY ./script_corrector.py /app

EXPOSE 9304

# RUN pip3 install --upgrade pip \
#     && pip3 install --no-cache-dir -r requirements.txt

# RUN pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install sentencepiece accelerate

CMD [ "flask", "--app=script_corrector.py", "run", "--host=0.0.0.0", "--port=6004" ]