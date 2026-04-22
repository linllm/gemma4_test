import os
import time
import uuid
import runpod
from vllm import LLM, SamplingParams

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-31B-it")

print(f"Loading model {MODEL_ID}...")
llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
    max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
    trust_remote_code=True,
)
print("Model loaded.")


def handler(job):
    input_data = job["input"]

    messages = input_data.get("messages")
    prompt = input_data.get("prompt")

    if messages:
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if not prompt:
        return {"error": "Provide 'prompt' or 'messages' in input."}

    params = SamplingParams(
        temperature=input_data.get("temperature", 0.7),
        top_p=input_data.get("top_p", 0.9),
        max_tokens=input_data.get("max_tokens", 512),
        repetition_penalty=input_data.get("repetition_penalty", 1.0),
    )

    outputs = llm.generate([prompt], params)
    generated = outputs[0].outputs[0].text
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(outputs[0].outputs[0].token_ids)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


runpod.serverless.start({"handler": handler})
