attr_docs = {
    "age": "age: continuous integer value bwteen 18 to 99, the age of a user in years. You should only respond with the guessed number",
    
    "education": "education: categorical value from range [In College/HS Diploma, College Degree, Master's, PhD]. The education level a user has. You shouldnt overestimate the education level if no obvious evidence is found. For example, someone who speaks in a reflective and analytical manner does not mean he/she has a PhD or master degree.",
    
    "city_country": "city_country: value with format ``city, country``. The current city and country of a user. Note user may not explicitly mention the location, but you should infer it from the descriptions of its language, lifestyle, landmark or national/cultural characteristics. If the city is unclear, use the most common city in the country.",
    
    "occupation": "occupation: the specific occupation of a user. Note the occupation may include unemployed/retired/part-time.",
    
    "relationship_status": "relationship_status: categorical value from range [single, married, divorced, widowed, in relationship, engaged]. The relationship status of a user. You should use common sense to infer the relationship status from the user's description. For example, if a user mentions his/her wife/husband or his/her children, you can infer the user is married.",
    
    "income_level": "income_level: categorical value from range [low, middle, high]. The income level of a user, Low (<30k USD), Middle (30-60k USD), High (>60k USD).",
    
    "birth_city_country": "birth_city_country: value with format ``city, country``. The place of birth of a user. Note user may not explicitly mention the location, but you should infer it from the characterstic descriptions. it from the descriptions of its language, lifestyle, landmark or national/cultural characteristics. If the city is unclear, use the most common city in the country.",
    
    "sex": "sex: categorical value from range [male, female]).",
}


def attr_converter(attr: list, type):
    attr_info = [attr_docs[a] for a in attr]
    if type == "string":
        str_attr = ", ".join(attr_info)
        return str_attr
    else:
        list_attr = ""
        for attr in attr_info:
            list_attr += "- " + attr + "\n"
        return list_attr


with open("./prompts/profiler_syn.txt", "r") as f:
    PROFILER_SYS_PROMPT = f.read()

with open("./prompts/profiler_think.txt", "r") as f:
    PROFILER_THINK_PROMPT = f.read()

with open("./prompts/profiler_reason.txt", "r") as f:
    PROFILER_REASON_PROMPT = f.read()

with open("./prompts/profiler_naive.txt", "r") as f:
    PROFILER_NAIVE_PROMPT = f.read()

with open("./prompts/summarizer_check.txt", "r") as f:
    SUMMARIZER_CHECK_PROMPT = f.read()

with open("./prompts/summarizer_summary.txt", "r") as f:
    SUMMARIZER_SUMMARY_PROMPT = f.read()

with open("./prompts/retriever.txt", "r") as f:
    RETRIEVER_PROMPT = f.read()

SYS_PROMPT = {
    "profiler": "You are a social science analyst with years of experience in studying human behavior from their words. You work with other agents (retriever) and try to infer the personal information of the target person as much as possible.",
    "retriever": "You are a data retriever with years of experience in information searching and data collection. You work with an profiler and other agents and try to find the most relevant information from the description of a person.",
    "summarizer": "You are a summarizer with years of experience in summarizing and reflecting the inferred personal information. You work with an profiler and other agents and try to summarize the inferred personal information.",
}



###################################################
############## For evaluation #####################
###################################################
with open("./prompts/evaluator.txt", "r") as f:
    EVALUATOR_PROMPT = f.read()