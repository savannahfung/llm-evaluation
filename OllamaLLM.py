import requests
import json
import time
import ollama
from LLM import LLM

class OllamaLLM(LLM):
    '''
    Ollama LLM class for creating and generating messages using the specified model and the given messages.

    Attributes:
        model (str): The model name.

    Note:
        This is using the Ollama API.
    '''
    def __init__(self, model):
        self.model = model

    def create(self, messages):
        try:
            res = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True
            )
        except ollama.ResponseError as e:
            print("Error:", e.error)
            if e.status_code == 404:
                ollama.pull(self.model)
        return res

    def generate(self, messages):
        start_time = time.time()
        stream = self.create(messages)
        res_content = ""
        for chunk in stream:
            res = chunk
            if res["done"] is True:
                end_time = time.time()
                break
            if res["message"]["role"] is not None:
                created_time = time.time()
                res_role = res["message"]["role"]
            if res["message"]["content"] is not None:
                res_content += res["message"]["content"]

        if res["done"] is True:
            res_status = "success"
        else:
            print("Error: response finished unexpectedly.")
            res_status = "error"

        response = {
            "status": res_status,
            "object": res
        }
        total_duration = end_time - start_time
        ttft = created_time - start_time
        tpot = total_duration / res["eval_count"] if res["eval_count"] > 0 else 0
        throughput = 1 / (ttft + tpot)
        res_message = {
            "role": res_role,
            "content": res_content
        }

        return {
            "ttft": ttft,
            "tpot": tpot,
            "throughput": throughput,
            "latency": total_duration,
            "message": res_message,
            "response": response
        }
