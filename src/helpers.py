import re
import os
import nbformat
import copy
from typing import Dict


class MessageRole:
    ASSISTANT = "assistant"
    USER = "user"
    PSEUDO_USER = "pseudo_user"
    GESTURE = "gesture"
    SYSTEM = "system"


def merge_adjacent_msgs_from_same_role(msg_list):
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


def extract_role_and_content_from_cell_source(cell: Dict):
    output_content_list = None
    line_content = ""
    cell_content = ""
    cell_source = cell['source'].strip()
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
        # raise Exception(f"Unknown Message Role from the content: {cell_source}")
    if role is None:
        cell_content = cell_source
    else:
        lines = cell_source.split("\n")

        # content = "\n".join([line for line in lines if not line.startswith(("%%", "#%%p", "# %%p", ))])
        for ith, line in enumerate(lines):
            if line.startswith(("%%", "#%%p", "# %%p", "%")) and ith == 0:
                line_actual = line.split()[-1]  # TODO: handle multiple spaces
                line_content += line_actual
            else:
                cell_content += line + "\n"

        cell_content = cell_content.strip()
        if role == MessageRole.PSEUDO_USER:
            output_content_list = parse_notebook_outputs_to_tool_outputs(cell['outputs'])
        if role == MessageRole.GESTURE:
            output_content_list = parse_notebook_outputs_to_tool_outputs(cell['outputs'])
            cell_content = output_content_list[0]['text']
            output_content_list = None

    # return role, content, output_content_list
    return {
        "role": role,
        "cell_content": cell_content,
        "line_content": line_content,
        "raw_cell": cell, 
        "output_content_list": output_content_list,
    }


def convert_cell_list_to_msg_list(cell_list, print_cell=False):
    system = None
    msg_list = []
    for ith, cell in enumerate(cell_list):
        result = extract_role_and_content_from_cell_source(cell)
        # role, content, output_content_list = extract_role_and_content_from_cell_source(cell)
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


def read_notebook_into_cell_jsons(notebook_path: str, print_cell=False):
    with open(notebook_path, "r") as fin:
        nb = nbformat.read(fin, as_version=4)
    return nb['cells']


def load_notebook_as_msg_list(notebook_path: str):
    cell_list = read_notebook_into_cell_jsons(notebook_path=notebook_path)
    msg_list, system = convert_cell_list_to_msg_list(cell_list)
    claude_msg_list = convert_msg_list_to_claude_msg_format(msg_list)
    claude_msg_list = merge_adjacent_msgs_from_same_role(claude_msg_list)
    if len(claude_msg_list) == 0:
        logger.warning(f"Empty Claude message list. Notebook path: {notebook_path}")
        return claude_msg_list, system

    # check the claude msg list alternates between user and assistant
    is_valid_claude_msg_list = True
    for i in range(len(claude_msg_list) - 1):
        if claude_msg_list[i]['role'] == claude_msg_list[i + 1]['role']:
            raise Exception(f"Invalid Claude message list at {i} location: {claude_msg_list[i:i+2]}")
            is_valid_claude_msg_list = False
            break
    if claude_msg_list[0]['role'] == MessageRole.ASSISTANT or claude_msg_list[-1]['role'] == MessageRole.USER:
        raise Exception(f"Invalid Claude message list. Message list shall start with role USER and end with ASSISTANT.\nactual start role: {claude_msg_list[0]['role']}. actual end role: {claude_msg_list[-1]['role']}")
        is_valid_claude_msg_list = False

    return claude_msg_list, system


def add_fewshots_from_path(path_str: str, extension=".ipynb"):
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
    for fp in files:
        nb_msg, system = load_notebook_as_msg_list(notebook_path=fp)
        few_shots.extend(nb_msg)
        # merged_msg_list_for_claude = [msg for msg_list in few_shots for msg in msg_list] + merged_msg_list_for_claude
    return few_shots, system


def extract_string_list_from_xml_tags(text, tag_name):
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    matches = list(re.findall(pattern, text, re.DOTALL))
    # if matches:
    #     matches = matches[0]
    return matches


def convert_msg_list_to_claude_msg_format(msg_list):
    claude_msg_list = []
    for msg in msg_list:
        new_msg = copy.deepcopy(msg)
        if new_msg['role'] == MessageRole.PSEUDO_USER or new_msg['role'] == MessageRole.GESTURE:
            new_msg['role'] = MessageRole.USER
        claude_msg_list.append(new_msg)
    return claude_msg_list
