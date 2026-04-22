FROM vllm/vllm-openai:latest

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY handler.py .

ENV HF_HOME=/runpod-volume/huggingface
ENV TENSOR_PARALLEL_SIZE=1
ENV MAX_MODEL_LEN=8192

CMD ["python", "-u", "handler.py"]
