FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir gdown

COPY . /app

# Descarga desde Google Drive
RUN gdown --id 1eB1LfjfJPdK6rNmbXb6A86Ptn-CSj1zz -O modelo_rf_actualizado.pkl

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}


