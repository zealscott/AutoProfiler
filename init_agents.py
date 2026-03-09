"""
this file is used to initialize the agents used for profiling attack
"""

from agents.retriever import Retriever
from agents.profiler import Profiler
from agents.summarizer import Summarizer
from agents.tagger import Tagger
from core.toolkit import ServiceToolkit

from functions.web import bing_search, digest_webpage
from config.web_api import GOOGLE_API_KEY, GOOGLE_ID, BING_API_KEY
from functions.local import get_new_history, get_related_history, get_all_history

from util.prompt_loader import SYS_PROMPT


def init_retriever(user_history, knowledge_base, visited_history, count_token, model_name="gpt-4o", api_key=None):
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
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    retriever = Retriever(
        name="retriever",
        model=model_name,
        count_token=count_token,
        service_toolkit=service_toolkit,
        sys_prompt=SYS_PROMPT["retriever"],
        **kwargs,
    )
    return retriever


def init_profiler(target_attributes, count_token=False, model_name="gpt-4o", api_key=None):
    """
    Initialize the profiler agent
    """
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    # Create agents
    profiler = Profiler(
        name="profiler",
        model=model_name,
        target_attributes=target_attributes,
        count_token=count_token,
        sys_prompt=SYS_PROMPT["profiler"],
        **kwargs,
    )
    return profiler


def init_tagger(model_name="gpt-4o", api_key=None):
    """Initialize the tagger agent."""
    from tagging import TAG_PROMPT

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    tagger = Tagger(
        name="tagger",
        model=model_name,
        sys_prompt=TAG_PROMPT,
        **kwargs,
    )
    return tagger


def init_summarizer(target_attributes, count_token, model_name="gpt-4o", api_key=None):
    """
    Initialize the summarizer agent
    """
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    # Create agents
    analyst = Summarizer(
        name="summarizer",
        model=model_name,
        target_attributes=target_attributes,
        count_token=count_token,
        sys_prompt=SYS_PROMPT["summarizer"],
        **kwargs,
    )
    return analyst
