# Large Language Models Evaluation

## Requirements

Python version: 3.10 (recommended)

## Setup

Set up a python virtual environment

```
python -m venv .venv
```

Clone git repository

```
git clone https://github.com/savannahfung/llm-evaluation.git
```

Install required packages

```
source .venv/bin/activate
cd llm-evaluation/
pip install -r requirements.txt
```

Install Ollama from https://ollama.ai/download

```
curl https://ollama.ai/install.sh | sh
```

## Quick Start

llm-evaluation allows you to evaluate large language mdels against different metrics with different API and libraries.

### API Options

1. OpenAI API
2. vLLM API Server
3. vLLM with OpenAI-Compatible Server
4. Ollama API

```
python evaluate.py
```

### Single Test

It runs the large language model with the input files once. It logs the results of the test cases into the output file and return the average metrics (including BLEU, Rouge-1, Rouge-L, TTFT, TPOT, Throughput, and Latency).

```
LLMEvaluator.evaluate_all()
```

### Simulation

It runs and sends concurrent requests to the large language model for each input file. It logs the results of the test cases into the output file and return the average metrics (including BLEU, Rouge-1, Rouge-L, TTFT, TPOT, Throughput, and Latency).

```
LLMEvaluator.simulate_user(num_users, avg_res_time)
```

### Metrics Graph

It sends concurrent requests to the large language model and increase the number of concurrent requests for each iteration until the maximum number of concurrent requests. It logs the metrics for each simulation and plot the graph for each metrics again the number of concurrent requests.

```
LLMEvaluator.metrics_graph(max_users, avg_res_time, step)
```

### Load Test

It sends concurrent requests to the large language model and increase the number of concurrent requests for each iteration until the maximum latency. It logs the metrics for each simulation and plot the graph for each metrics again the number of concurrent requests.

```
LLMEvaluator.load_test(max_latency, avg_res_time, step)
```

## Bugs

- vLLM with OpenAI-Compatible Server hangs as number of concurrent requests increase.
- Ollama API sometimes only use CPU to run large language models.
- vllmLLM class and vLLM API is currently not fully functioning. It only supports completion.

## FAQs

### How to set OpenAI api key and organization id?

Set OPENAI_API_KEY and OPENAI_ORG_ID variables in the .env file.

### Where can I find the server logs?

Server logs can be found in the server_logs/ directory.
