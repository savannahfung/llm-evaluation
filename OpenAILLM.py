from openai import OpenAI
import time
from LLM import LLM

class OpenAILLM(LLM):
    '''
    OpenAI LLM class for creating and generating messages using the specified model and the given messages.

    Attributes:
        model (str): The model name.
        client (OpenAI): The OpenAI client.

    Note:
        If using OpenAI API, API key must be provided.
        If using vLLM with OpenAI-compatible server, the base URL must be provided.
    '''
    client = None

    def __init__(self, model, api_key = "EMPTY", organization_id = None, base_url = "https://api.openai.com/v1"):
        self.model = model
        self.client = OpenAI(
            api_key = api_key,
            organization = organization_id,
            base_url = base_url
        )

    def create(self, messages):
        res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        return res

    def generate(self, messages):
        start_time = time.time()
        stream = self.create(messages)
        res_content = ""
        for chunk in stream:
            res = chunk
            if res.choices[0].finish_reason is not None:
                end_time = time.time()
                break
            if res.choices[0].delta.role is not None:
                created_time = time.time()
                res_role = res.choices[0].delta.role
            if res.choices[0].delta.content is not None:
                res_content += res.choices[0].delta.content
            
        if res.choices[0].finish_reason == "stop":
            res_status = "success"
        elif res.choices[0].finish_reason == "length":
            print("Warning: response reached max tokens.")
            res_status = "max_tokens"
        else:
            print("Error: response finished unexpectedly.")
            res_status = "error"

        response = {
            "status": res_status,
            "object": res
        }
        total_duration = end_time - start_time
        ttft = created_time - start_time
        tpot = total_duration / res.usage["completion_tokens"] if res.usage["completion_tokens"] > 0 else 0
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
