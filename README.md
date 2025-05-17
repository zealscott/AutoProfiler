## Automatic Profile Inference Attack on Synthetic Dataset

This is the implementation for AutoProfiler on the synthetic dataset (SynthPAI). 


### Code structure
* `/agent` contains the implementation of LLM agent for the attack.
* `/config` contains the API keys for various LLM providers.
* `/dataset` contains the orginal user history (`/synthpai`) as well as the inference results of each LLM on each user. 
* `/functions` contains the function callings for retriever.
* `/prompts` contains all prompts for each agent.
* `/util` contains scripts about data loader and useful functions.


### Dataset

We use the dataset and ground truth from the paper: `A Synthetic Dataset for Personal Attribute Inference`. 



### Attack
Simply run the script for inference attack:
```shell
python main.py -m [llm_model] -u [user]
```
This will load three agents (i.e., Retriever, Profiler, and Summarizer) to complete the task. 

The results are stored at `/dataset/{llm_model}/`. 

Note that in implementation we combine Strategist and Extractor as Profiler.

