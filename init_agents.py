"""
this file is used to initialize the agents used for profiling attack
"""

from agents.retriever import Retriever
from agents.profiler import Profiler
from agents.summarizer import Summarizer
from agentscope.service import (
    google_search,
    digest_webpage,
    ServiceToolkit,
    bing_search,
)

from agentscope.utils.token_utils import count_openai_token

from config.web_api import GOOGLE_API_KEY, GOOGLE_ID, BING_API_KEY
from functions.local import get_new_history, get_related_history, get_all_history

from util.prompt_loader import SYS_PROMPT


def init_retriever(user_history, knowledge_base, visited_history, count_token):
    """
    Initialize the retriever agent
    """
    # Prepare the tools for the agent
    service_toolkit = ServiceToolkit()
    # service_toolkit.add(google_search, api_key=GOOGLE_API_KEY, cse_id=GOOGLE_ID, num_results=20)
    service_toolkit.add(bing_search, api_key=BING_API_KEY, num_results=20)
    service_toolkit.add(digest_webpage)
    service_toolkit.add(get_all_history, synthpai_history=user_history)
    service_toolkit.add(get_new_history, synthpai_history=user_history, visited=visited_history, n=5)
    service_toolkit.add(get_related_history, synthpai_history=user_history, knowledge=knowledge_base, top_k=5)

    # Create agents
    retriever = Retriever(
        name="retriever",
        model_config_name="api_config",
        count_token=False,
        service_toolkit=service_toolkit,
        sys_prompt=SYS_PROMPT["retriever"],
    )
    return retriever


def init_profiler(target_attributes, count_token=False):
    """
    Initialize the profiler agent
    """
    # Create agents
    profiler = Profiler(
        name="profiler",
        model_config_name="api_config",
        target_attributes=target_attributes,
        count_token=False,
        sys_prompt=SYS_PROMPT["profiler"],
    )
    return profiler


def init_summarizer(target_attributes, count_token):
    """
    Initialize the summarizer agent
    """
    # Create agents
    analyst = Summarizer(
        name="summarizer",
        model_config_name="api_config",
        target_attributes=target_attributes,
        count_token=False,
        sys_prompt=SYS_PROMPT["summarizer"],
    )
    return analyst
