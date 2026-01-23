import json
import time
import os
import random
import copy
from typing import List, Dict
import typer
from loguru import logger
from tqdm import tqdm
import multiprocessing
from functools import partial

from utils import (
    set_random_seed,
    format_schema_to_markdown,
    create_simple_message,
    write_jsonl,
    read_jsonl_file,
    standardize_amb_unans_category,
)
from fewshot_utils import (
    add_fewshots_from_path,
    extract_string_list_from_xml_tags,
)
from litellm_helpers import (
    get_litellm_router,
    convert_claude_msg_list_to_litellm_msg_list,
    router_completion_with_ratelimit_retry,
)


MAX_OUTPUT_TOKENS = 4096
MIXTRAL_MAX_OUTPUT_TOKENS = 4096
CLAUDE_MAX_OUTPUT_TOKENS = 4096
LLAMA_MAX_OUTPUT_TOKENS = 2048

CURRENT_DIR = os.path.dirname(__file__)

set_random_seed(7)


def flatten_interleave(nested_list, k=None, truncate_to_min_len=True):
    if k == 0:
        return []

    if k % 2 != 0:
        raise Exception(f"This function is used for flattening fewshots examples and k can only be even number. Current k: {k}")
    flattened = []
    min_len = min(len(sublist) for sublist in nested_list)
    max_len = max(len(sublist) for sublist in nested_list)
    max_len = min(max_len, k) if k else max_len

    if truncate_to_min_len:
        nested_list = [sublist[:min_len] for sublist in nested_list]

    for i in range(0, max_len, 2):
        for sublist in nested_list:
            if i + 2 <= len(sublist):
                flattened.extend(sublist[i:(i + 2)])

    return flattened


def print_message_list(msg_list: List[Dict]):
    out_str_list = []
    for msg in msg_list:
        out_str_list.append(
            f"**{msg['role']}**:\n\n{msg['content']}"
        )
        out_str_list.append("---" * 30)
    return "\n".join(out_str_list)


def get_binary_classification_user_msg(schema_preview: str, initial_question: str, category: str, fewshots: List, system: str, litellm=True):
    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n<question>\n{initial_question}\n</question>"
    # logger.info(f"User message:\n{user_msg}")
    system_as_msg = {
        "role": "system",
        "content": system
    }
    tmp_msgs = [system_as_msg] + fewshots + [create_simple_message(message=user_msg, role="user")]
    if litellm:
        tmp_msgs = convert_claude_msg_list_to_litellm_msg_list(tmp_msgs)
    return tmp_msgs


def get_all_category_classification_user_msg(schema_preview: str, initial_question: str, category: str, fewshots: List, system: str, litellm=True):
    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n<question>\n{initial_question}\n</question>"
    system_as_msg = {
        "role": "system",
        "content": system
    }
    tmp_msgs = [system_as_msg] + fewshots + [create_simple_message(message=user_msg, role="user")]
    if litellm:
        tmp_msgs = convert_claude_msg_list_to_litellm_msg_list(tmp_msgs)
    return tmp_msgs


# @cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def perform_binary_classify_single_line(line: Dict, llm_model: str, fewshots_mapping: Dict, system_mapping: Dict, cell_value_key: str, litellm_router):
    conversation = line['finalConversation']
    initial_question = conversation[0]['USER']
    all_cell_values = line['retrievedCellValues'][cell_value_key]
    category = line['ambiguousUnanswerableCategory']
    schema_preview = format_schema_to_markdown(schema=all_cell_values, num_sample_cell_to_show=4)
    schema_mod = copy.deepcopy(line['schemaModification'])
    schema_mod.pop("LLM_Based_Alternative_Columns_To_Remove", None)
    # logger.info(f"Category: {category}\nSchema Modification:\n{json.dumps(schema_mod, indent=2)}")
    # logger.info(f"Category: {category}\nSchema Modification:\n{schema_mod}")

    result_key = f"binaryClassificationResult___{llm_model}___{cell_value_key}"

    if category == "Ambiguous_Values_Within_Column":
        fewshots_nested_list = [fewshots_mapping[category], fewshots_mapping[f"{category}_negative"]]
        system = """
You are a Database Expert (DB EXPERT) system that classifies when a user question is ambiguous due to multiple similar values within a column or not.

You will receive:
1. A database schema in markdown format with relevant column values enclosed in <schema/> tags.
2. The user's question enclosed in <question/> tags.

You shall classify the question as either "Ambiguous_Values_Within_Column" or "Other".

"Ambiguous_Values_Within_Column" means that multiple similar values in a column match the mentioned value, leading to multiple valid SQL queries that differ in the specific filtering cell value from one column in the WHERE clause.
"Other" means all questions that do not meet the definition of "Ambiguous_Values_Within_Column", such as questions with a nonexistent filter value (the mentioned filter value is not present in the schema and no SQL can be constructed with the appropriate value in the WHERE clause) or answerable questions (unambiguous and can be answered with the given data).

An question is answerable if and only if both of the following condtions are met:
- the question posed is unambiguous, precise, and leaving no room for multiple interpretations or confusion. 
- the database contains the complete set of information required to formulate a comprehensive and accurate response to the query.

Note that we shall output "Ambiguous_Values_Within_Column" if there is any possible ambiguities in the filter value in the question given the database schema.

Provide your step-by-step thoughts within </scratch> tags.
Then, provide your final classification within <result/> tags as one of the two categories above ("Ambiguous_Values_Within_Column" or "Other"), without any extra explanation.
"""
    else:
        fewshots_nested_list = [fewshots_mapping[category], fewshots_mapping['answerable']]
        system = """
You are a Database Expert (DB EXPERT) system that classifies user questions into one of the following two categories based on the given database schema:
{category_with_explanation}
- answerable: the database contains data needed to answer the question and the question has one and only one valid interpreation.

You will receive:
1. A database schema in markdown format with relevant column values enclosed in <schema/> tags.
2. The user's question enclosed in <question/> tags.

Note that the "answerable" output shall only be provided if and only if:
- the question posed is unambiguous, precise, and leaving no room for multiple interpretations or confusion. 
- the database contains the complete set of information required to formulate a comprehensive and accurate response to the query. 
If either of these conditions is not met, meaning the question lacks clarity or our data is insufficient, we shall refrain from classifying the query as "answerable."

Provide your step-by-step thoughts within </scratch> tags.
Then, provide your final classification within <result/> tags as one of the categories above. Note that you result shall only be one of the categories specified at the beginning & Do not include any extra explanation in the result. 
""".format(
    category_with_explanation=system_mapping[category],
)
    random.shuffle(fewshots_nested_list)
    if "claude" in llm_model:
        fewshots = flatten_interleave(fewshots_nested_list, k=8, truncate_to_min_len=True)
        max_output_tokens = CLAUDE_MAX_OUTPUT_TOKENS
    elif "llama3" in llm_model:
        fewshots = flatten_interleave(fewshots_nested_list, k=8, truncate_to_min_len=True)
        max_output_tokens = LLAMA_MAX_OUTPUT_TOKENS
    elif "mixtral" in llm_model:
        fewshots = flatten_interleave(fewshots_nested_list, k=2, truncate_to_min_len=True)
        max_output_tokens = MIXTRAL_MAX_OUTPUT_TOKENS

    if category == "answerable":
        fewshots = fewshots_mapping["answerable"][:4]

    tmp_msgs = get_binary_classification_user_msg(
        schema_preview=schema_preview,
        initial_question=initial_question,
        category=category,
        fewshots=fewshots,
        system=system,
    )

    # logger.info(f"System prompt:\n{system}")
    # logger.info(f"Input Messages:\n{print_message_list(tmp_msgs)}")

    try:
        # response_obj = litellm_router.completion(
        #     model=llm_model,
        #     messages=tmp_msgs,
        #     temperature=0.0,
        #     top_p=1,
        #     max_tokens=max_output_tokens,
        # )
        response_obj = router_completion_with_ratelimit_retry(
            model=llm_model,
            messages=tmp_msgs,
            router=litellm_router,
            temperature=0.0,
            top_p=1,
            max_tokens=max_output_tokens,
        )
        line['binaryClassificationExceptionRawResponseObject'] = str(response_obj)
    except Exception as ex:
        logger.error(f"Completion error with message: {ex}")
        line[result_key] = None
        line["binaryClassificationException"] = str(ex)
        return line

    try:
        response = response_obj.choices[0].message.content
        result = extract_string_list_from_xml_tags(response, "result")[0].strip()
        # logger.info(f"Response:\n{response}")
        # print("===" * 30)
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        try:
            logger.error(f"Raw response obj: {response_obj}")
        except Exception as ex222:  # noqa:
            pass
        response = None
        result = None

    line[result_key] = {
        "inputPrompt": system,
        "completeMessages": tmp_msgs,
        "raw": response,
        "parsed": result
    }
    return copy.deepcopy(line)


# @cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def perform_all_type_classify_single_line(line: Dict, llm_model: str, fewshots_mapping: Dict, system_mapping: Dict, cell_value_key: str, k_shots: int, litellm_router):
    conversation = line['finalConversation']
    initial_question = conversation[0]['USER']
    all_cell_values = line['retrievedCellValues'][cell_value_key]
    category = line['ambiguousUnanswerableCategory']
    schema_preview = format_schema_to_markdown(schema=all_cell_values, num_sample_cell_to_show=4)
    schema_mod = copy.deepcopy(line['schemaModification'])
    schema_mod.pop("LLM_Based_Alternative_Columns_To_Remove", None)
    # logger.info(f"Category: {category}\nSchema Modification:\n{json.dumps(schema_mod, indent=2)}")
    # logger.info(f"Category: {category}\nSchema Modification:\n{schema_mod}")
    result_key = f"allCategoryClassificationResult___{llm_model}___{cell_value_key}"

    system_mapping = copy.deepcopy(system_mapping)
    system_mapping.pop("answerable", None)
    all_category_explanation = []
    for cat, exp in system_mapping.items():
        # tmp_row = f"- {cat} : exp"
        tmp_row = exp
        all_category_explanation.append(tmp_row)
    all_category_explanation_str = "\n".join(all_category_explanation)

    fewshots_nested_list = list(fewshots_mapping.values())
    random.shuffle(fewshots_nested_list)
    if "claude" in llm_model:
        fewshots = flatten_interleave(fewshots_nested_list, k=k_shots * 2, truncate_to_min_len=True)
        max_output_tokens = CLAUDE_MAX_OUTPUT_TOKENS
    elif "llama3" in llm_model:
        fewshots = flatten_interleave(fewshots_nested_list, k=k_shots * 2, truncate_to_min_len=True)
        max_output_tokens = LLAMA_MAX_OUTPUT_TOKENS
    elif "mixtral" in llm_model:
        fewshots = flatten_interleave(fewshots_nested_list, k=k_shots * 2, truncate_to_min_len=True)
        max_output_tokens = MIXTRAL_MAX_OUTPUT_TOKENS

    system = """
You are a Database Expert (DB EXPERT) system that classifies user questions into one of the following 9 categories based on the given database schema:
{category_with_explanation}
- answerable: the database contains data needed to answer the question and the question has one and only one valid interpreation.

You will receive:
1. A database schema in markdown format with relevant column values enclosed in <schema/> tags.
2. The user's question enclosed in <question/> tags.

Your output should follow this format:
<scratch> YOUR-STEP-BY-STEP-THOUGHTS </scratch>
<result> ONE-OF-THE-9-QUESTION-CATEGORIES </result>

Note that the "answerable" output shall only be provided if and only if:
- the question posed is unambiguous, precise, and leaving no room for multiple interpretations or confusion. 
- the database contains the complete set of information required to formulate a comprehensive and accurate response to the query. 
If either of these conditions is not met, meaning the question lacks clarity or our data is insufficient, we shall refrain from classifying the query as "answerable."

Provide your step-by-step thoughts within </scratch> tags.
Then, provide your final classification within <result/> tags as one of the categories above. Do not include any extra explanation in the result.
""".format(
    category_with_explanation=all_category_explanation_str,
)

    tmp_msgs = get_all_category_classification_user_msg(
        schema_preview=schema_preview,
        initial_question=initial_question,
        category=category,
        fewshots=fewshots,
        system=system,
    )
    # logger.info(f"System prompt:\n{system}")
    # logger.info(f"Input Messages:\n{print_message_list(tmp_msgs)}")

    try:
        start_time = time.time()
        # response_obj = litellm_router.completion(
        #     model=llm_model,
        #     messages=tmp_msgs,
        #     temperature=0.0,
        #     top_p=1,
        #     max_tokens=max_output_tokens,
        # )
        response_obj = router_completion_with_ratelimit_retry(
            model=llm_model,
            messages=tmp_msgs,
            router=litellm_router,
            temperature=0.0,
            top_p=1,
            max_tokens=max_output_tokens,
        )
        line['allCategoryClassificationExceptionRawResponseOjbect'] = str(response_obj)
        end_time = time.time()
        time_lapse = end_time - start_time
    except Exception as ex:
        end_time = time.time()
        time_lapse = end_time - start_time
        logger.error(f"Time lapsed since initial call: {time_lapse}\nCompletion error with message: {ex}")
        line[result_key] = None
        line["allCategoryClassificationException"] = str(ex)
        return line

    try:
        response = response_obj.choices[0].message.content
        result = extract_string_list_from_xml_tags(response, "result")[0].strip()
        # logger.info(f"Response:\n{response}")
        # print("===" * 30)
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        try:
            logger.error(f"Raw response: {response_obj}")
            line['allCategoryClassificationException'] = str(ex) + str(response_obj)
        except Exception as ex222:  # noqa:
            pass
            line['allCategoryClassificationException'] = str(ex)
        response = None
        result = None

    line[result_key] = {
        "inputPrompt": system,
        "completeMessages": tmp_msgs,
        "raw": response,
        "parsed": result
    }
    return copy.deepcopy(line)


def main(
    infp: str = typer.Option(
        ..., help="Input JSONL file path (combined data file)"
    ),
    classification_type: str = typer.Option(
        "binary", help="Classification type: 'binary' (category vs Other) or 'all-category' (classify across all 9 categories)"
    ),
    llm_model: str = typer.Option(
        "claude-3-5-sonnet", help="LLM model to use for classification (e.g., claude-3-5-sonnet, claude-3-sonnet, llama3-1-70b)"
    ),
    cell_value_key: str = typer.Option(
        "lexicalAndOracle", help="Cell value type to use: 'lexicalAndOracle' or 'lexicalOnly'"
    ),
    k_shots: int = typer.Option(
        3, help="Number of few-shot examples per category (only for all-category classification)"
    ),
    num_processes: int = typer.Option(
        12, help="Number of parallel processes for classification"
    ),
):
    """
    Perform binary or all-category classification on combined ambiguous/unanswerable data.

    This optional step validates that generated examples truly belong to their assigned category,
    which helps improve data quality before the final contextualization step.
    """

    # Category definitions
    category_to_explanation_mapping = {
        "Ambiguous_SELECT_Column": "Multiple columns match the requested output information, leading to multiple valid SQLs that differ in the columns used in the SELECT clause.",
        "Ambiguous_WHERE_Column": "The filter condition matches multiple columns in a table, leading to multiple valid SQLs that differ in the specific filter column in the WHERE clause.",
        "Ambiguous_Values_Within_Column": "Multiple similar values in a column match the mentioned value, leading to multiple valid SQLs that differ in the specific filtering cell value from one column in the WHERE clause.",
        "Ambiguous_Filter_Criteria": "The question contains a filter condition or criteria that is ambiguous, vague, relative/descriptive, or open to multiple interpretations. This ambiguity in the filter criteria makes it difficult to formulate a precise SQL query without additional clarification.",
        "Nonexistent_SELECT_Column": "At least one of the requested output information is not present in the schema, so no SQL can be constructed with the appropriate column in the SELECT clause.",
        "Nonexistent_WHERE_Column": "At least one filter condition column is not present in the schema, so no SQL can be constructed with the appropriate column in the WHERE clause",
        "Unsupported_Join": "The required join between tables is not supported due to a lack of common columns, preventing the construction of a valid SQL query",
        "Nonexistent_Filter_Value": "The mentioned filtering value is not present in the schema, so no SQL can be constructed with the appropriate value in the WHERE clause.",
    }

    category_to_fewshots_fp_mapping = {
        "Ambiguous_SELECT_Column": "src/classification_fewshots_binary/ambiguous_select_column.ipynb",
        "Ambiguous_WHERE_Column": "src/classification_fewshots_binary/ambiguous_where_column.ipynb",
        "Ambiguous_Values_Within_Column": "src/classification_fewshots_binary/ambiguous_values_within_column.ipynb",
        "Ambiguous_Values_Within_Column_negative": "src/classification_fewshots_binary/ambiguous_values_within_column_negative.ipynb",
        "Ambiguous_Filter_Criteria": "src/classification_fewshots_binary/ambiguous_filter_criteria.ipynb",
        "Nonexistent_SELECT_Column": "src/classification_fewshots_binary/nonexistent_select_column.ipynb",
        "Nonexistent_WHERE_Column": "src/classification_fewshots_binary/nonexistent_where_column.ipynb",
        "Unsupported_Join": "src/classification_fewshots_binary/unsupported_join.ipynb",
        "Nonexistent_Filter_Value": "src/classification_fewshots_binary/nonexistent_filter_value.ipynb",
        "answerable": "src/classification_fewshots_binary/answerable.ipynb",
    }

    # Load few-shot examples
    category_to_fewshots_msg = {}
    for key, val in category_to_fewshots_fp_mapping.items():
        absolute_fp = os.path.join(CURRENT_DIR, "..", val)
        category_to_fewshots_msg[key], _ = add_fewshots_from_path(absolute_fp)

    category_to_category_with_explanation_mapping = {}
    for key, val in category_to_explanation_mapping.items():
        category_to_category_with_explanation_mapping[key] = f"- {key}: {val}"
    category_to_category_with_explanation_mapping['answerable'] = ""

    # Load input data
    logger.info(f"Loading data from: {infp}")
    lines = read_jsonl_file(infp)

    # Standardize category names
    for line in lines:
        line = standardize_amb_unans_category(line)

    all_unique_categories = list(set([line['ambiguousUnanswerableCategory'] for line in lines]))
    all_unique_categories_str = '\n'.join(all_unique_categories)
    logger.info(f"Found {len(lines)} examples across {len(all_unique_categories)} categories:\n{all_unique_categories_str}")

    # Validate categories
    for category in all_unique_categories:
        if category not in category_to_fewshots_fp_mapping:
            raise Exception(f"{category} not in the mapping: {category_to_fewshots_msg.keys()}")

    # Initialize router
    router = get_litellm_router()

    logger.info(f"Classification settings:")
    logger.info(f"  Type: {classification_type}")
    logger.info(f"  Model: {llm_model}")
    logger.info(f"  Cell value key: {cell_value_key}")
    logger.info(f"  K-shots: {k_shots} (for all-category only)")
    logger.info(f"  Parallel processes: {num_processes}")

    # Perform classification
    if classification_type == "binary":
        logger.info("Performing binary classification (category vs Other)...")
        with multiprocessing.Pool(processes=num_processes) as pool:
            processed_lines = list(tqdm(pool.imap(
                partial(
                    perform_binary_classify_single_line,
                    llm_model=llm_model,
                    fewshots_mapping=category_to_fewshots_msg,
                    system_mapping=category_to_category_with_explanation_mapping,
                    cell_value_key=cell_value_key,
                    litellm_router=router
                ),
                lines
            ), total=len(lines)))
        out_fp = f"{infp}.binary_classification___{llm_model}___{cell_value_key}.jsonl"

    elif classification_type == "all-category":
        logger.info("Performing all-category classification (9 categories)...")
        with multiprocessing.Pool(processes=num_processes) as pool:
            processed_lines = list(tqdm(pool.imap(
                partial(
                    perform_all_type_classify_single_line,
                    llm_model=llm_model,
                    fewshots_mapping=category_to_fewshots_msg,
                    system_mapping=category_to_category_with_explanation_mapping,
                    cell_value_key=cell_value_key,
                    k_shots=k_shots,
                    litellm_router=router
                ),
                lines
            ), total=len(lines)))
        out_fp = f"{infp}.all_category_classification___{llm_model}___{cell_value_key}___shots_{k_shots}.jsonl"

    else:
        raise ValueError(f"Invalid classification_type: {classification_type}. Must be 'binary' or 'all-category'")

    # Write output
    logger.info(f"Writing output to: {out_fp}")
    write_jsonl(processed_lines, out_fp)
    logger.info(f"Classification complete! Output saved to: {out_fp}")


if __name__ == '__main__':
    typer.run(main)
