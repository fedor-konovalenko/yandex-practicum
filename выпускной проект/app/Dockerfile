FROM python:3.10.1-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y libgomp1
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
