import threading
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import time
import OpenAILLM
import OllamaLLM
import vllmLLM

class LLMEvaluator:
    '''
    LLMEvaluator class for evaluating the LLM model using the given input and reference files.

    Attributes:
        llm (LLM): The LLM object.
        model_name (str): The model name.
        input_base (str): The base directory for input files.
        reference_base (str): The base directory for reference files.
        output_base (str): The base directory for output files.
        simulation_output_base (str): The base directory for simulation output files.
        plots_base (str): The base directory for plots.
        metrics_base (str): The base directory for metrics.
        simulation_metrics_base (str): The base directory for simulation metrics.
        files (list): The list of input files.

    Note:
        The input files must be in JSON format. Each line is a JSON object with the following fields: role, content.
        The reference files must be in plain text format. Each line is a message.
    '''
    llm = None
    model_name = None
    input_base = "./input/"
    reference_base = "./reference/"
    output_base = "./results/output/"
    simulation_output_base = "./results/output/simulation/"
    plots_base = "./results/plots/"
    metrics_base = "./results/metrics/"
    simulation_metrics_base = "./results/metrics/simulation/"
    files = []

    def __init__(self, llm):
        self.llm = llm
        self.files = [f for f in os.listdir(self.input_base) if os.path.isfile(os.path.join(self.input_base, f))]
        self.model_name = llm.model.split("/")[-1]
        self.output_base += self.model_name + "/"
        self.simulation_output_base += self.model_name + "/"
        self.plots_base += self.model_name + "/"
        os.makedirs(self.output_base, exist_ok=True)
        os.makedirs(self.simulation_output_base, exist_ok=True)
        os.makedirs(self.plots_base, exist_ok=True)
        os.makedirs(self.metrics_base, exist_ok=True)
        os.makedirs(self.simulation_metrics_base, exist_ok=True)

    def evaluate(self, input_file, output_file, reference_file, user_id = None, avg_res_time = None, results = None):
        chat_history = []
        with open(input_file, "r") as inf, open(output_file, "w") as outf, open(reference_file, "r") as ref:
            if user_id is not None:
                outf.write("User ID: " + str(user_id) + "\n")
            outf.write("Model: " + self.llm.model + "\n")
            outf.write("Input file: " + input_file + "\n\n")
            metrics = {
                "bleu": 0,
                "rouge1": 0,
                "rougeL": 0,
                "ttft": 0,
                "tpot": 0,
                "throughput": 0,
                "latency": 0,
            }

            total_lines = sum(1 for line in inf)
            inf.seek(0)
            desc = f"{'User ' + str(user_id) + ': ' if user_id is not None else ''}Processing {input_file.split('/')[-1]}"
            for line_num, (input_line, ref_line) in enumerate(zip(tqdm(inf, total=total_lines, desc=desc, unit="lines"), ref)):
                message = json.loads(input_line)
                chat_history.append(message)

                if message["role"] == "system":
                    outf.write("> " + message["content"] + "\n\n")
                    continue

                if avg_res_time is not None:
                    res_time = np.random.weibull(1) / avg_res_time
                    time.sleep(res_time)

                res = self.llm.generate(chat_history)
                chat_history.append(res["message"])

                smoothing = SmoothingFunction().method1
                bleu = sentence_bleu([ref_line.split()], res["message"]["content"].split(), smoothing_function=smoothing)
                metrics["bleu"] += bleu

                scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                rouge = scorer.score(ref_line, res["message"]["content"])
                metrics["rouge1"] += rouge["rouge1"].fmeasure
                metrics["rougeL"] += rouge["rougeL"].fmeasure

                metrics["ttft"] += res["ttft"]
                metrics["tpot"] += res["tpot"]
                metrics["throughput"] += res["throughput"]
                metrics["latency"] += res["latency"]

                outf.write(">>> " + message["content"] + "\n\n")
                outf.write(res["message"]["content"] + "\n\n")
                outf.write("BLEU: " + str(bleu) + "\n")
                outf.write("Rouge-1: " + str(rouge["rouge1"].fmeasure) + "\n")
                outf.write("Rouge-L: " + str(rouge["rougeL"].fmeasure) + "\n")
                outf.write("TTFT: " + str(res["ttft"]) + "\n")
                outf.write("TPOT: " + str(res["tpot"]) + "\n")
                outf.write("Throughput: " + str(res["throughput"]) + "\n")
                outf.write("Latency: " + str(res["latency"]) + "\n\n")

                if res["response"]["status"] != "success":
                    outf.write("Stop request.\n")
                    outf.write("Status: " + res["response"]["status"] + "\n")
                    outf.write("Object: " + str(res["response"]["object"]) + "\n")
                    break
            
            metrics["bleu"] /= (total_lines - 1)
            metrics["rouge1"] /= (total_lines - 1)
            metrics["rougeL"] /= (total_lines - 1)
            metrics["ttft"] /= (total_lines - 1)
            metrics["tpot"] /= (total_lines - 1)
            metrics["throughput"] /= (total_lines - 1)
            metrics["latency"] /= (total_lines - 1)

            outf.write("Average BLEU: " + str(metrics["bleu"]) + "\n")
            outf.write("Average Rouge-1: " + str(metrics["rouge1"]) + "\n")
            outf.write("Average Rouge-L: " + str(metrics["rougeL"]) + "\n")
            outf.write("Average TTFT: " + str(metrics["ttft"]) + "\n")
            outf.write("Average TPOT: " + str(metrics["tpot"]) + "\n")
            outf.write("Average Throughput: " + str(metrics["throughput"]) + "\n")
            outf.write("Average Latency: " + str(metrics["latency"]) + "\n")

            if results is not None:
                results.append(metrics)

            return metrics

    def evaluate_all(self):
        metrics = {
            "bleu": 0,
            "rouge1": 0,
            "rougeL": 0,
            "ttft": 0,
            "tpot": 0,
            "throughput": 0,
            "latency": 0,
        }
        total_files = len(self.files)
        metrics_file = self.metrics_base + self.model_name + ".txt"

        with open(metrics_file, "w") as mf:
            mf.write("Model: " + self.llm.model + "\n\n")
            for file_num, file in tqdm(enumerate(self.files, start=1), total=total_files, desc="Processing files", unit="files"):
                input_file = self.input_base + file
                reference_file = self.reference_base + file
                output_file = self.output_base + file
            
                res = self.evaluate(input_file, output_file, reference_file)
                
                metrics["bleu"] += res["bleu"]
                metrics["rouge1"] += res["rouge1"]
                metrics["rougeL"] += res["rougeL"]
                metrics["ttft"] += res["ttft"]
                metrics["tpot"] += res["tpot"]
                metrics["throughput"] += res["throughput"]
                metrics["latency"] += res["latency"]

                mf.write("File: " + file + "\n")
                mf.write("BLEU: " + str(res["bleu"]) + "\n")
                mf.write("Rouge-1: " + str(res["rouge1"]) + "\n")
                mf.write("Rouge-L: " + str(res["rougeL"]) + "\n")
                mf.write("TTFT: " + str(res["ttft"]) + "\n")
                mf.write("TPOT: " + str(res["tpot"]) + "\n")
                mf.write("Throughput: " + str(res["throughput"]) + "\n")
                mf.write("Latency: " + str(res["latency"]) + "\n\n")

            metrics["bleu"] /= total_files
            metrics["rouge1"] /= total_files
            metrics["rougeL"] /= total_files
            metrics["ttft"] /= total_files
            metrics["tpot"] /= total_files
            metrics["throughput"] /= total_files
            metrics["latency"] /= total_files

            mf.write("Average BLEU: " + str(metrics["bleu"]) + "\n")
            mf.write("Average Rouge-1: " + str(metrics["rouge1"]) + "\n")
            mf.write("Average Rouge-L: " + str(metrics["rougeL"]) + "\n")
            mf.write("Average TTFT: " + str(metrics["ttft"]) + "\n")
            mf.write("Average TPOT: " + str(metrics["tpot"]) + "\n")
            mf.write("Average Throughput: " + str(metrics["throughput"]) + "\n")
            mf.write("Average Latency: " + str(metrics["latency"]) + "\n")

            return metrics

    def simulate_user(self, num_users, avg_res_time):
        metrics = {
            "bleu": 0,
            "rouge1": 0,
            "rougeL": 0,
            "ttft": 0,
            "tpot": 0,
            "throughput": 0,
            "latency": 0,
        }
        total_files = len(self.files)
        metrics_file = self.simulation_metrics_base + self.model_name + ".txt"
        with open(metrics_file, "w") as mf:
            mf.write("Model: " + self.llm.model + "\n\n")
            mf.write("Number of users: " + str(num_users) + "\n")
            mf.write("Average response time: " + str(avg_res_time) + "\n\n")
            for file_num, file in tqdm(enumerate(self.files, start=1), total=total_files, desc="Processing files",
                                       unit="files"):
                input_file = self.input_base + file 
                reference_file = self.reference_base + file
                file_name = file.split(".")[0]
                output_file = self.simulation_output_base + file_name + "/"
                os.makedirs(output_file, exist_ok=True)
                fmetrics = {
                    "bleu": 0,
                    "rouge1": 0,
                    "rougeL": 0,
                    "ttft": 0,
                    "tpot": 0,
                    "throughput": 0,
                    "latency": 0,
                }

                threads = []
                results = []
                for user_id in range(num_users):
                    usr_output_file = output_file + str(user_id) + ".txt"
                    thread = threading.Thread(target=self.evaluate,
                                              args=(input_file, usr_output_file, reference_file, user_id, avg_res_time, results))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                for res in results:
                    fmetrics["bleu"] += res["bleu"]
                    fmetrics["rouge1"] += res["rouge1"]
                    fmetrics["rougeL"] += res["rougeL"]
                    fmetrics["ttft"] += res["ttft"]
                    fmetrics["tpot"] += res["tpot"]
                    fmetrics["throughput"] += res["throughput"]
                    fmetrics["latency"] += res["latency"]

                fmetrics["bleu"] /= num_users
                fmetrics["rouge1"] /= num_users
                fmetrics["rougeL"] /= num_users
                fmetrics["ttft"] /= num_users
                fmetrics["tpot"] /= num_users
                fmetrics["throughput"] /= num_users
                fmetrics["latency"] /= num_users

                metrics["bleu"] += fmetrics["bleu"]
                metrics["rouge1"] += fmetrics["rouge1"]
                metrics["rougeL"] += fmetrics["rougeL"]
                metrics["ttft"] += fmetrics["ttft"]
                metrics["tpot"] += fmetrics["tpot"]
                metrics["throughput"] += fmetrics["throughput"]
                metrics["latency"] += fmetrics["latency"]

                mf.write("File: " + file + "\n")
                mf.write("BLEU: " + str(fmetrics["bleu"]) + "\n")
                mf.write("Rouge-1: " + str(fmetrics["rouge1"]) + "\n")
                mf.write("Rouge-L: " + str(fmetrics["rougeL"]) + "\n")
                mf.write("TTFT: " + str(fmetrics["ttft"]) + "\n")
                mf.write("TPOT: " + str(fmetrics["tpot"]) + "\n")
                mf.write("Throughput: " + str(fmetrics["throughput"]) + "\n")
                mf.write("Latency: " + str(fmetrics["latency"]) + "\n\n")

            metrics["bleu"] /= total_files
            metrics["rouge1"] /= total_files
            metrics["rougeL"] /= total_files
            metrics["ttft"] /= total_files
            metrics["tpot"] /= total_files
            metrics["throughput"] /= total_files
            metrics["latency"] /= total_files

            mf.write("Average BLEU: " + str(metrics["bleu"]) + "\n")
            mf.write("Average Rouge-1: " + str(metrics["rouge1"]) + "\n")
            mf.write("Average Rouge-L: " + str(metrics["rougeL"]) + "\n")
            mf.write("Average TTFT: " + str(metrics["ttft"]) + "\n")
            mf.write("Average TPOT: " + str(metrics["tpot"]) + "\n")
            mf.write("Average Throughput: " + str(metrics["throughput"]) + "\n")
            mf.write("Average Latency: " + str(metrics["latency"]) + "\n")

            return metrics

    def plot_metrics(self, users, metrics):
        for key in metrics:
            plt.plot(users, metrics[key])
            plt.title(key)
            plt.xlabel("Number of users")
            plt.ylabel(key)
            file = self.plots_base + key + ".png"
            if os.path.isfile(file):
                os.remove(file)
            plt.savefig(file)
            plt.clf()

    def metrics_graph(self, max_user, avg_res_time, step):
        metrics = {
            "bleu": [],
            "rouge1": [],
            "rougeL": [],
            "ttft": [],
            "tpot": [],
            "throughput": [],
            "latency": [],
        }

        users = []
        for num_users in tqdm(range(1, max_user + 1, step), desc="Number of Users", unit="users", position=0):
            res = self.simulate_user(num_users, avg_res_time)
            users.append(num_users)
            for key in metrics:
                metrics[key].append(res[key])

        self.plot_metrics(users, metrics)

        return metrics

    def load_test(self, max_latency, avg_res_time, step):
        metrics = {
            "bleu": [],
            "rouge1": [],
            "rougeL": [],
            "ttft": [],
            "tpot": [],
            "throughput": [],
            "latency": [],
        }
        users = []
        curr_latency = 0
        num_users = 1

        latency_pbar = tqdm(total=max_latency, desc="Latency Progress", unit="latency", position=0)
        while curr_latency < max_latency:
            res = self.simulate_user(num_users, avg_res_time)
            users.append(num_users)
            for key in metrics:
                metrics[key].append(res[key])
            curr_latency = res["latency"]
            num_users += step
            latency_pbar.update(curr_latency)

        latency_pbar.close()
        num_users -= step
        print("Final number of users: " + str(num_users))
        print("Final latency: " + str(curr_latency))
        self.plot_metrics(users, metrics)

        return {
            "num_users": num_users,
            "final_latency": curr_latency,
            "metrics": metrics
        }