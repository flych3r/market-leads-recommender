FROM python:3.7

WORKDIR /app

COPY . .

EXPOSE 8080

RUN pip install -r requirements.txt

CMD cd src && streamlit run --server.port 8080 --server.enableCORS false App.py
