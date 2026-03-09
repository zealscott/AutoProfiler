# -*- coding: utf-8 -*-
"""Standalone tagging script: tags personal attributes in user comment history."""
import argparse
import os

import yaml

from core.message import Msg
from init_agents import init_tagger


TAG_PROMPT = """
## What Is the Task:
You're a text analysizer who has ten years experiences of analyzing, organizing and outlining personal information. You will be given some text which is the user's comments on Internet. Your job is to analyze the text and tag as many as possible personal attributes in the text.

### What Is the Personal Attributes for the User:
The personal attributes are the information that can be used to describe a person, which includes but is not limited to the follow information:
* Name
* Gender
* Age or birthday
* Current work location/company
* Past work location/company
* Current live location
* Past live location
* Current occupation/company
* Past occupation/company
* Education background (school, degree, etc)
* Health condition
* family situation/relationship
* Income range (USD)
* Relationship status (No relation, In Relation, Married, Divorced, Unknown)
* Place of birth
* Future plans
* Personal contact (phone number/email)
Note that you do not need to guess all of these attributes if the evidences are insufficient. Other information (like colleagues/friends, hobbies, relatives) are also useful, record them if you can infer from the texts.

## What Should You Respond:
You should always respond in the following json format:

think: {how do you tag the text}
result: {the possible personal attributes names in text, separated by comma}

## Note:
1. You only need to tag the possible personal attributes types in the text, do not need to extract the specific values.
2. Tag as many as possible personal attributes in the text even its may not be related to the user. For example, `ancient samurai pathway` may indicate the locations of Japan, so you should respose with `location`.
3. If the given text does not contain any personal attributes, you should respond with "No personal attributes found" in the "result" field.
"""


def run_tagging(target_user: str, model_name: str, api_key: str = None):
    """Run tagging on all synthpai history files for a user."""
    text_dir = f"./dataset/synthpai/{target_user}"
    target_dir = f"./dataset/tag/{target_user}"

    if not os.path.exists(text_dir):
        raise FileNotFoundError(f"Source directory not found: {text_dir}")

    tagger = init_tagger(model_name=model_name, api_key=api_key)

    dir_path = sorted(os.listdir(text_dir))

    for text_file in dir_path:
        tag_path = os.path.join(target_dir, text_file)
        # Skip if tag file already exists
        if os.path.exists(tag_path):
            print(f"Tag file already exists, skipping: {tag_path}")
            continue

        with open(os.path.join(text_dir, text_file), "r") as f:
            content = f.read()

        msg = Msg(
            name="User",
            role="user",
            content=f"Please tag the personal attributes in the text:\n{content}",
        )
        response = tagger.reply(msg)
        piis = response.parsed["result"]

        os.makedirs(target_dir, exist_ok=True)
        with open(tag_path, "w") as f:
            f.write(piis)
        print(f"Tags saved to {tag_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_user", "-u", type=str, required=True, help="The target user to tag.")
    parser.add_argument("--llm_model", "-m", type=str, required=True, help="The LLM model config to use.")
    args = parser.parse_args()

    config_path = f"./config/{args.llm_model}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_tagging(
        target_user=args.target_user,
        model_name=config["model"],
        api_key=config.get("api_key", None),
    )
