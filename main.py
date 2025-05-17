# -*- coding: utf-8 -*-
import json
import os
from agentscope.models import load_model_by_config_name
from agents.claude_agent import ClaudePostAPIChatWrapper
from agents.gemini_agent import GeminiPostAPIChatWrapper
from agentscope.rag import LlamaIndexKnowledge
from agentscope.message import Msg
import agentscope

import time

from init_agents import init_retriever, init_profiler, init_summarizer
from util.data_loader import load_synthpai, check_valid
from util.data_clean import deduplicate

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--target_user", "-u", type=str, help="The target user to infer PIIs.")
parser.add_argument("--llm_model", "-m", type=str, help="The LLM model to use.")

args = parser.parse_args()
target_user = args.target_user
llm_model = args.llm_model


# check whether the target user has ground truth
target_attributes = check_valid(target_user)
if target_attributes == []:
    exit()


model_config_path = f"./config/{llm_model}.json"
agentscope.init(model_configs=model_config_path)

rag_config = {
    "emb_model_config_name": "embedding_config",
    "data_processing": [
        {
            "load_data": {
                "loader": {
                    "create_object": True,
                    "module": "llama_index.core",
                    "class": "SimpleDirectoryReader",
                    "init_args": {
                        "input_dir": f"./dataset/tag/{target_user}",
                        "required_exts": [".txt"],
                    },
                }
            }
        },
    ],
}

emb_model = load_model_by_config_name("embedding_config")


knowledge = LlamaIndexKnowledge(
    knowledge_id=target_user, emb_model=emb_model, knowledge_config=rag_config, persist_root="./dataset/vdb"
)

# Load reddit history
user_history = load_synthpai(target_user)
visited_history = []

count_token = True if llm_model == "gpt-4" else False
retriever = init_retriever(user_history, knowledge, visited_history, count_token)
profiler = init_profiler(target_attributes, count_token)
summarizer = init_summarizer(target_attributes, count_token)


x = Msg(name="user", role="user", content="Please begin inferring\n")
# Build


key_piis = []  # we explicitly store the highly confident PIIs in case of forget
cur_piis = []  # this serves as the compression of memory

while True:
    time.sleep(5)
    instruct_msg = profiler.think(x, reset=True)

    while instruct_msg.parsed["action"] == "retrieval" or instruct_msg.parsed["action"] == "search":
        # send the message to the retriever
        instruct_msg = Msg(name="profiler", role="assistant", content=instruct_msg.parsed["instruction"])
        data_msg = retriever(instruct_msg)
        # send the response to the profiler
        instruct_msg = profiler.think(data_msg)

    if instruct_msg.parsed["action"] == "finish":
        if len(visited_history) == len(user_history):
            # force to reason agine with all user history
            res_msg = profiler.naive_infer(target_attributes, user_history)
            new_piis = res_msg.parsed["results"]
            key_piis.extend(new_piis)
            # final check with inferred and once-inferred PIIs
            final_msg = Msg(name="profiler", role="assistant", content=key_piis)
            # update the inferred PIIs
            final_piis = summarizer.check(final_msg).parsed["results"]
            final_piis = deduplicate(final_piis)
            break
        else:
            x = Msg(
                name="user",
                role="user",
                content=f"You already infer: {x}. However, there is still more user's comment history, you should retrieval more for reasoning. Keep going!\n",
            )

    if instruct_msg.parsed["action"] == "reason":
        ########### start reasoning ###########
        instruct_msg = Msg(name="profiler", role="assistant", content=instruct_msg.parsed["instruction"])
        res_msg = profiler.reason(instruct_msg)
        ########### get the inferred PIIs ###########
        new_piis = res_msg.parsed["results"]
        ####### store highly confident PIIs in case of forget #######
        for attr_dict in new_piis:
            # Check if attr_dict is actually a dictionary
            try:
                if attr_dict["type"] in target_attributes:
                    # if the attribute is the target attribute, store it
                    print(f"Stored target attribute: {attr_dict}")
                    key_piis.append(attr_dict)
            except:
                print(f"Error: {attr_dict} is not a dictionary")
        # combine all cur_piis
        cur_piis.extend(new_piis)
        ########### check the inferred PIIs ###########
        # send the response to the summarizer for checking
        res_msg = Msg(name="profiler", role="assistant", content=cur_piis)
        # update the inferred PIIs
        cur_piis = summarizer.check(res_msg).parsed["results"]
        # perpare for the next iteration
        x = Msg(name="Summarizer", role="assistant", content=cur_piis)


print(f"Inferred PIIs: {json.dumps(final_piis, indent=2)}")

print("The summary of the inferred PIIs:")
res_msg = Msg(name="user", role="user", content=final_piis)
description = summarizer.summary(res_msg)
print(description.parsed["summary"])


# save the inferred PIIs
os.makedirs(f"./dataset/{llm_model}/pii/", exist_ok=True)
os.makedirs(f"./dataset/{llm_model}/summary/", exist_ok=True)

with open(f"./dataset/{llm_model}/pii/{target_user}.json", "w") as f:
    json.dump(final_piis, f, indent=2)

with open(f"./dataset/{llm_model}/summary/{target_user}.txt", "w") as f:
    if description.parsed["summary"] == None:
        f.write("No summary available")
    else:
        f.write(description.parsed["summary"])

if len(final_piis) != len(target_attributes):
    print("The inferred PIIs are not complete!")
    with open(f"./incomplete_{llm_model}.txt", "w+") as f:
        f.write(target_user)
