import sys
import io

from agentscope.service import (
    ServiceResponse,
    ServiceExecStatus,
)

from agentscope.rag import llama_index_knowledge


def get_new_history(synthpai_history: list, visited: list, n: int = 5) -> ServiceResponse:
    """
    Retrieve the next user's synthetic history from the given list of histories in reverse chronological order.
    This is the default function for user history retrieval when no specific requirements are given.
    Args:
        synthpai_history (`list`):
            The list of user synthetic histories.
        visited (`list`):
            The list of visited history indices.
        n (`int`, defaults to `5`):
            The number of histories to retrieve.
    Returns:
        `ServiceResponse`: A dictionary with two variables: `status` and
        `content`. The `status` variable is from the ServiceExecStatus enum,
        and `content` is a string of user synthetic histories or error information,
        which depends on the `status` variable.
    """
    if len(visited) == len(synthpai_history):
        return ServiceResponse(
            ServiceExecStatus.SUCCESS, "All histories have been visited. No new history to retrieve."
        )

    cursor = visited[-1] + 1 if visited else 0
    # Retrieve the user synthetic history
    output = ""
    for i in range(cursor, min(cursor + n, len(synthpai_history))):
        output += synthpai_history[i] + "\n"

    # Update the visited list
    visited.extend(range(cursor, min(cursor + n, len(synthpai_history))))

    return ServiceResponse(ServiceExecStatus.SUCCESS, output)


def get_related_history(
    synthpai_history: list, query: str, knowledge: llama_index_knowledge, top_k=5
) -> ServiceResponse:
    """
    Retrieve the semantic related user synthetic history based on the given query.
    Args:
        query (`str`):
            The query to retrieve the related user synthetic history.
        knowledge (`llama_index_knowledge`):
            The knowledge object that contains the user synthetic history.
        top_k (`int`, defaults to `5`):
            The number of related histories to retrieve.
    Returns:
        `ServiceResponse`: A dictionary with two variables: `status` and
        `content`. The `status` variable is from the ServiceExecStatus enum,
        and `content` is a string of related user synthetic histories or error information,
        which depends on the `status` variable.
    """
    # Retrieve the related user synthetic history
    try:
        output = []
        nodes = knowledge.retrieve(query, similarity_top_k=top_k, to_list_strs=False)
        for node in nodes:
            filename = node.metadata["file_name"]
            idx = int(filename.strip(".txt").split("_")[-1])
            output.append(synthpai_history[idx - 1])
        return ServiceResponse(ServiceExecStatus.SUCCESS, output)
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))


def get_all_history(synthpai_history: list) -> ServiceResponse:
    """
    Retrieve ALL history from the given list of histories in reverse chronological order.
    This is only used when the user wants to retrieve all histories.
    Args:
        synthpai_history (`list`):
            The list of user synthetic histories.
    Returns:
        `ServiceResponse`: A dictionary with two variables: `status` and
        `content`. The `status` variable is from the ServiceExecStatus enum,
        and `content` is a string of user synthetic histories or error information,
        which depends on the `status` variable.
    """
    # Retrieve the user synthetic history
    output = ""
    for i in range(len(synthpai_history)):
        output += synthpai_history[i] + "\n"

    return ServiceResponse(ServiceExecStatus.SUCCESS, output)
