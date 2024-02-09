import requests
import json
import time
from LLM import LLM

class vllmLLM(LLM):
    '''
    vLLM LLM class for creating and generating messages using the specified model and the given messages.

    Attributes:
        model (str): The model name.
        base_url (str): The base URL of the server.

    Note:
        This is using the vLLM API.
        This class is currently not fully functioning.
    '''
    model = None

    def __init__(self, model):
        self.model = model

    def create(self, messages):
        res = requests.post(
            "http://localhost:8000/generate",
            json = {
                "prompt": messages,
                "stream": True
            },
            stream = True
        )
        return res

    def generate(self, messages):
        num_tokens = 0
        start_time = time.time()
        stream = self.create(messages)
        res = None
        for chunk in stream.iter_lines(delimiter=b"\0"):
            if chunk:
                end_time = time.time()
                num_tokens += 1
                if num_tokens == 1:
                    created_time = time.time()
                res = json.loads(chunk)

        if res is not None:
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
        tpot = total_duration / num_tokens if num_tokens > 0 else 0
        throughput = 1 / (ttft + tpot)
        res_message = res["text"].split("</assistant>")[-2].res["text"]("<assistant>")[-1]

        return {
            "ttft": ttft,
            "tpot": tpot,
            "throughput": throughput,
            "latency": total_duration,
            "message": res_message,
            "response": response
        }
           