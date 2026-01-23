"""
Utilities for loading few-shot examples from Jupyter notebooks and parsing XML tags from LLM responses.

This module provides functions to:
1. Load few-shot examples from .ipynb files for LLM prompting
2. Parse XML-tagged content from LLM responses
3. Convert notebook cells to message format for LLM APIs
"""

import nbformat
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
import os
import re
import copy

# Import the new unified LLM interface
from llm_interface import get_default_llm


# For backwards compatibility, create a default LLM instance
# This replaces the old BEDROCK_LLM with the new unified interface
BEDROCK_LLM = get_default_llm()

# Constant used in notebook parsing
CLICKABLE_OUTPUT_TAG = "details"


class MessageRole:
    """Message roles used in notebook cell parsing."""
    ASSISTANT = "assistant"
    USER = "user"
    PSEUDO_USER = "pseudo_user"
    GESTURE = "gesture"
    SYSTEM = "system"


def extract_string_list_from_xml_tags(text: str, tag_name: str) -> List[str]:
    """
    Extract content from XML tags in text.

    Args:
        text: The text containing XML tags
        tag_name: The tag name to extract (without < >)

    Returns:
        List of strings found within the specified tags

    Example:
        >>> extract_string_list_from_xml_tags("<result>value1</result><result>value2</result>", "result")
        ['value1', 'value2']
    """
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    matches = list(re.findall(pattern, text, re.DOTALL))
    return matches


def read_notebook_into_cell_jsons(notebook_path: str, print_cell: bool = False) -> List[Dict]:
    """
    Read a Jupyter notebook file and return its cells as a list of dictionaries.

    Args:
        notebook_path: Path to the .ipynb file
        print_cell: Unused parameter, kept for backwards compatibility

    Returns:
        List of cell dictionaries from the notebook
    """
    with open(notebook_path, "r") as fin:
        nb = nbformat.read(fin, as_version=4)
    return nb['cells']


def parse_notebook_outputs_to_tool_outputs(cell_outputs: List[Dict]) -> List[Dict]:
    """
    Parse notebook cell outputs into a format suitable for LLM tool outputs.

    Handles:
    - stdout/stderr streams
    - Images (PNG)
    - HTML outputs
    - Markdown outputs
    - Error tracebacks

    Args:
        cell_outputs: List of output dictionaries from a notebook cell

    Returns:
        List of formatted output content dictionaries
    """
    output_content_list = []
    for output in cell_outputs:
        if output['output_type'] == 'stream':
            if output['name'] == 'stdout':
                output_content_list.append(
                    {
                        "type": "text",
                        "text": f"<stdout>\n{output['text']}\n</stdout>",
                    }
                )
        elif output['output_type'] == 'display_data' and output['data'].get("image/png"):
            output_content_list.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": output['data'].get("image/png"),
                    }
                }
            )
        elif output['output_type'] == "error":
            output_content_list.append(
                {
                    "type": "text",
                    "text": f"<stderr>\n{output['traceback']}\n</stderr>",
                }
            )
        elif output['output_type'] == "execute_result" and output['data'].get("text/html"):
            output_content_list.append(
                {
                    "type": "text",
                    "text": f"<stdout>\n{output['data']['text/html']}\n</stdout>",
                }
            )
        elif output['output_type'] == "display_data" and output['data'].get("text/markdown"):
            tmp_text = output['data']['text/markdown']
            # Try to extract content from clickable tags if present
            extracted = extract_string_list_from_xml_tags(tmp_text, CLICKABLE_OUTPUT_TAG)
            if extracted:
                tmp_text = extracted[0]
            output_content_list.append(
                {
                    "type": "text",
                    "text": tmp_text,
                }
            )
        else:
            logger.error(f"Unknown output type: {output['output_type']}. Output: {output}")
    return output_content_list


def extract_role_and_content_from_cell_source(cell: Dict) -> Dict[str, Any]:
    """
    Extract role and content from a notebook cell based on magic commands.

    Recognizes the following cell prefixes:
    - %%system or %system: System message
    - %%u or %u: User message
    - %%upload or %upload: Gesture (file upload)
    - %%a: Assistant message
    - #%%p or # %%p or %%p: Pseudo-user message

    Args:
        cell: Dictionary representing a notebook cell

    Returns:
        Dictionary containing:
        - role: The message role (from MessageRole class)
        - cell_content: The main content text
        - line_content: Content from the magic line
        - raw_cell: The original cell dictionary
        - output_content_list: Parsed outputs (if any)
    """
    output_content_list = None
    line_content = ""
    cell_content = ""
    cell_source = cell['source'].strip()

    # Determine role based on cell prefix
    if cell_source.startswith(("%system", "%%system")):
        role = MessageRole.SYSTEM
    elif cell_source.startswith(("%%u", "%u")):
        if cell_source.startswith(("%%upload", "%upload")):
            role = MessageRole.GESTURE
        else:
            role = MessageRole.USER
    elif cell_source.startswith("%%a"):
        role = MessageRole.ASSISTANT
    elif cell_source.startswith(("#%%p", "# %%p", "%%p")):
        role = MessageRole.PSEUDO_USER
    else:
        role = None

    if role is None:
        cell_content = cell_source
    else:
        lines = cell_source.split("\n")

        # Extract content, skipping magic command line
        for ith, line in enumerate(lines):
            if line.startswith(("%%", "#%%p", "# %%p", "%")) and ith == 0:
                line_actual = line.split()[-1]  # TODO: handle multiple spaces
                line_content += line_actual
            else:
                cell_content += line + "\n"

        cell_content = cell_content.strip()

        # Parse outputs for certain roles
        if role == MessageRole.PSEUDO_USER:
            output_content_list = parse_notebook_outputs_to_tool_outputs(cell['outputs'])
        if role == MessageRole.GESTURE:
            output_content_list = parse_notebook_outputs_to_tool_outputs(cell['outputs'])
            if output_content_list:
                cell_content = output_content_list[0]['text']
            output_content_list = None

    return {
        "role": role,
        "cell_content": cell_content,
        "line_content": line_content,
        "raw_cell": cell,
        "output_content_list": output_content_list,
    }


def merge_adjacent_msgs_from_same_role(msg_list: List[Dict]) -> List[Dict]:
    """
    Merge consecutive messages with the same role.

    Claude API requires alternating user/assistant messages, so this function
    merges adjacent messages from the same role into a single message.

    Args:
        msg_list: List of message dictionaries with 'role' and 'content' keys

    Returns:
        List of merged message dictionaries
    """
    if len(msg_list) < 1:
        return msg_list

    merged_msg_list = [msg_list[0]]
    for ith in range(1, len(msg_list)):
        msg = msg_list[ith]
        prev_msg = merged_msg_list[-1]
        if prev_msg['role'] == msg['role']:
            prev_msg['content'].extend(msg['content'])
        else:
            merged_msg_list.append(msg)
    return merged_msg_list


def convert_msg_list_to_claude_msg_format(msg_list: List[Dict]) -> List[Dict]:
    """
    Convert message list to Claude API format.

    Converts PSEUDO_USER and GESTURE roles to USER role for Claude compatibility.

    Args:
        msg_list: List of message dictionaries

    Returns:
        List of message dictionaries in Claude format
    """
    claude_msg_list = []
    for msg in msg_list:
        new_msg = copy.deepcopy(msg)
        if new_msg['role'] == MessageRole.PSEUDO_USER or new_msg['role'] == MessageRole.GESTURE:
            new_msg['role'] = MessageRole.USER
        claude_msg_list.append(new_msg)
    return claude_msg_list


def convert_cell_list_to_msg_list(cell_list: List[Dict], print_cell: bool = False) -> Tuple[List[Dict], Optional[str]]:
    """
    Convert a list of notebook cells to a message list suitable for LLM APIs.

    Args:
        cell_list: List of cell dictionaries from a notebook
        print_cell: If True, print role and content for debugging

    Returns:
        Tuple of (message_list, system_prompt)
        - message_list: List of message dictionaries with role and content
        - system_prompt: System message content if present, otherwise None
    """
    system = None
    msg_list = []

    for ith, cell in enumerate(cell_list):
        result = extract_role_and_content_from_cell_source(cell)
        role, content, output_content_list = result['role'], result['cell_content'], result['output_content_list']

        if print_cell:
            print(role, content)

        if role:
            if role == MessageRole.SYSTEM:
                if system is None:
                    system = content
                else:
                    raise Exception(f"System message already exist. It is defined again in cell: {content}")
            else:
                if content.strip():
                    msg = {
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    }
                    if output_content_list:
                        msg['content'].extend(output_content_list)
                    msg_list.append(msg)

    return msg_list, system


def load_notebook_as_msg_list(notebook_path: str) -> Tuple[List[Dict], Optional[str]]:
    """
    Load a Jupyter notebook and convert it to a message list for LLM APIs.

    This function:
    1. Reads the notebook cells
    2. Converts cells to messages based on magic commands
    3. Converts to Claude format
    4. Merges adjacent messages from the same role
    5. Validates the message list

    Args:
        notebook_path: Path to the .ipynb file

    Returns:
        Tuple of (message_list, system_prompt)

    Raises:
        Exception: If the message list is invalid (non-alternating roles or wrong start/end roles)
    """
    cell_list = read_notebook_into_cell_jsons(notebook_path=notebook_path)
    msg_list, system = convert_cell_list_to_msg_list(cell_list)
    claude_msg_list = convert_msg_list_to_claude_msg_format(msg_list)
    claude_msg_list = merge_adjacent_msgs_from_same_role(claude_msg_list)

    if len(claude_msg_list) == 0:
        logger.warning(f"Empty Claude message list. Notebook path: {notebook_path}")
        return claude_msg_list, system

    # Validate the claude msg list alternates between user and assistant
    for i in range(len(claude_msg_list) - 1):
        if claude_msg_list[i]['role'] == claude_msg_list[i + 1]['role']:
            raise Exception(f"Invalid Claude message list at {i} location: {claude_msg_list[i:i+2]}")

    # Check that message list starts with USER and ends with ASSISTANT
    if claude_msg_list[0]['role'] == MessageRole.ASSISTANT or claude_msg_list[-1]['role'] == MessageRole.USER:
        raise Exception(
            f"Invalid Claude message list. Message list shall start with role USER and end with ASSISTANT.\n"
            f"actual start role: {claude_msg_list[0]['role']}. actual end role: {claude_msg_list[-1]['role']}"
        )

    return claude_msg_list, system


def add_fewshots_from_path(path_str: str, extension: str = ".ipynb") -> Tuple[List[Dict], Optional[str]]:
    """
    Load few-shot examples from notebook file(s).

    This is the main function used by generation scripts to load few-shot examples.
    It can handle either a single file or a directory of files.

    Args:
        path_str: Path to a .ipynb file or directory containing .ipynb files
        extension: File extension to filter (default: ".ipynb")

    Returns:
        Tuple of (few_shot_messages, system_prompt)
        - few_shot_messages: List of message dictionaries from all notebooks
        - system_prompt: System prompt from the last notebook (if any)

    Raises:
        Exception: If path_str is neither a file nor a directory
    """
    files = []
    if os.path.isfile(path_str):
        files.append(path_str)
    elif os.path.isdir(path_str):
        for fn in os.listdir(path_str):
            if fn.endswith(extension):
                files.append(os.path.join(path_str, fn))
    else:
        raise Exception(f"Unknown path: {path_str}")

    few_shots = []
    system = None
    for fp in files:
        nb_msg, system = load_notebook_as_msg_list(notebook_path=fp)
        few_shots.extend(nb_msg)

    return few_shots, system
