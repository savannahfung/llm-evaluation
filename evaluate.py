from dotenv import load_dotenv
import os
import subprocess
import ollama
import torch
from server import start_server
from OpenAILLM import OpenAILLM
from OllamaLLM import OllamaLLM
from vllmLLM import vllmLLM
from LLMEvaluator import LLMEvaluator

def main():
    input_base = "./input/"
    reference_base = "./reference/"
    os.makedirs(input_base, exist_ok=True)
    os.makedirs(reference_base, exist_ok=True)

    print("1. OpenAI API")
    print("2. vLLM API Server")
    print("3. vLLM with OpenAI-Compatible Server")
    print("4. Ollama API")

    try:
        while True:
            user_input = input("Enter the number: ")
            if user_input == "1":
                model = input("Enter the model name: ")
                load_dotenv(".env")
                llm = OpenAILLM(model, api_key=os.getenv("OPENAI_API_KEY"), organization_id=os.getenv("OPENAI_ORG_ID"))
                break

            elif user_input == "2":
                model = input("Enter the model name: ")
                num_gpus = int(input("Enter the number of GPUs: "))
                command = f"python -m vllm.entrypoints.api_server --model {model} --tensor-parallel-size {num_gpus}"
                server = start_server(command, 8000)
                llm = vllmLLM(model)
                break

            elif user_input == "3":
                model = input("Enter the model name: ")
                num_gpus = int(input("Enter the number of GPUs: "))
                command = f"python -m vllm.entrypoints.openai.api_server --model {model} --tensor-parallel-size {num_gpus}"
                server = start_server(command, 8000)
                llm = OpenAILLM(model, base_url="http://localhost:8000/v1")
                break

            elif user_input == "4":
                model = input("Enter the model name: ")
                command = f"ollama serve"
                server = start_server(command, 11434)
                try:
                    if ollama.pull(model)['status'] == 'success':
                        llm = OllamaLLM(model)
                        break
                    else:
                        print("Error: pull failed.")
                        continue
                except Exception as e:
                    print(f"Error: {e}")
                    continue

            else:
                print("Invalid input.")
                continue

        evaluator = LLMEvaluator(llm)

        user_input = input("Run single test? (y/n) ")
        if user_input == "y":
            res = evaluator.evaluate_all()
            print("Results:")
            print(res)
            print()
        
        user_input = input("Run simulation? (y/n) ")
        if user_input == "y":
            num_users = int(input("Number of users: "))
            avg_res_time = float(input("Average response time (sec): "))
            res = evaluator.simulate_user(num_users, avg_res_time)
            print("Results:")
            print(res)
            print()

        user_input = input("Plot metrics graph? (y/n) ")
        if user_input == "y":
            max_users = int(input("Max users: "))
            avg_res_time = float(input("Average response time (sec): "))
            step = int(input("Step: "))
            res = evaluator.metrics_graph(max_users, avg_res_time, step)
            print("Results:")
            print(res)
            print()

        user_input = input("Run load test? (y/n) ")
        if user_input == "y":
            max_latency = float(input("Max latency (sec): "))
            avg_res_time = float(input("Average response time (sec): "))
            step = int(input("Step: "))
            res = evaluator.load_test(max_latency, avg_res_time, step)
            print("Results:")
            print(res)
            print()
        
    except Exception as e:
        print(f"Error: {e}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()