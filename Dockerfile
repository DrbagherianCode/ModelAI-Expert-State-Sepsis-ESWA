FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends wget libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/modelai
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "run_all.py"]
