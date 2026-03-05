FROM chambie0102/chamber-xray-base:latest

WORKDIR /app

# Only copy training script — deps + weights already in base
COPY train.py .

CMD ["python", "train.py"]
