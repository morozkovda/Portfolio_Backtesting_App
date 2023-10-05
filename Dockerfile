FROM python:3.10-slim-bullseye
RUN apt-get update && apt-get install -y cmake
WORKDIR /app
COPY ../../Downloads/portfolio_backtest-main /app
RUN pip install -r requirements.txt
ENV NAME Portfolio_Management
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
CMD ["streamlit","run", "app.py"]
