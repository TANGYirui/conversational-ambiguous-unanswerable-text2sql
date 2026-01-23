import pandas as pd 
import json 
import os
import random
import copy
from typing import List, Dict, Tuple
import typer
from loguru import logger
from tqdm import tqdm
import multiprocessing
from functools import partial

from simple_cache import (
    cache_results
)

from custom_sql_engine import (
    DbWithModification,
    MISSING_CELL_VALUE,
)
from utils import (
    set_random_seed,
    format_schema_to_markdown,
    create_simple_message,
    read_jsonl_file,
    get_select_table_column_info,
    get_lexically_similar_cell_values_from_schema,
    get_lexically_similar_columns_from_schema,
    load_spider_dev_train_data,
    get_all_table_column_info,
    sample_questions_by_database,
    parse_for_where,
    is_numeric,
    write_jsonl,
)
from helpers import (
    add_fewshots_from_path,
    extract_string_list_from_xml_tags,
)
from litellm_helpers import (
    get_litellm_router,
    convert_claude_msg_list_to_litellm_msg_list,
)


AMB_UNANS_CATEGORY = "Ambiguous_VALUES_within_Column"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "__cache__")  # use current dir as a cache dire
os.makedirs(CACHE_DIR, exist_ok=True)
IGNORE_CACHE = True
CURRENT_DIR = os.path.dirname(__file__)

set_random_seed(7)


@cache_results(CACHE_DIR, ignore_cache=IGNORE_CACHE)
def get_sql_cell_filter(line, spider_root_dir: str = ""):
    where_tuples = parse_for_where(line['query'])
    text_tuples = []
    spider_database_dir = os.path.join(spider_root_dir, "database")
    for where_tuple in where_tuples:
        is_non_numeric_or_id = (
            not is_numeric(where_tuple['value']) or "id" in where_tuple['column'].lower()
        )
        if where_tuple['operator'] == "=" and is_non_numeric_or_id:
            text_tuples.append(where_tuple)
    if text_tuples:
        single_where_clause = random.choice(text_tuples)
        line['ambiguousUnanswerableCategory'] = AMB_UNANS_CATEGORY
        line['schemaModification'] = {
            "removeCell": [
                {
                    "table": single_where_clause["table"], "column": single_where_clause["column"],
                    "operator": single_where_clause['operator'], "value": single_where_clause['value']
                }
            ]
        }
        return line
    else:
        if "where" in line['query'].lower():
            logger.debug(f"No where clause: {line['query']}")
        # logger.debug(f"No where clause: {data['query']}")
        return None


@cache_results(CACHE_DIR, ignore_cache=IGNORE_CACHE)
def identify_additional_cell_values_to_remove(line, spider_root_dir, system_prompt="", few_shots=[], litellm_router=None):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    primary_cell_to_remove = line['schemaModification']['removeCell'][0]
    # if len(parsed_select_clause) > 1:
    #     logger.debug(f"More than one select clause: {line['query']}")
    if primary_cell_to_remove:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=line['schemaModification'])
        tab_col_cells = online_db.get_cell_values(only_unique_value=True)
        lexically_related_cells = get_lexically_similar_cell_values_from_schema(primary_cell_to_remove, tab_col_cells=tab_col_cells)
        line['ambiguousUnanswerableCategory'] = AMB_UNANS_CATEGORY
        lexically_related_cells_update = {
            "removeCellLexicallyRelated": lexically_related_cells
        }
        line['schemaModification'].update(lexically_related_cells_update)
        return line
    else:
        logger.info(f"No cells to remove from current line. question: {line['question']}. Schema modification: {line['schemaModification']}")
        return None


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def generate_replacement_cell_values_for_single_line(
    line, spider_root_dir, 
    system_prompt, few_shots,
    litellm_router=None
):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    schema_modification = line.get("schemaModification", {})

    # if cell value is numeric, no need to generate ambiguous values
    if is_numeric(schema_modification['removeCell'][0]['value']):
        return None

    try:
        # we show the complete schema to help model find better replacement columns that will be ambiguous
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification={})
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return {}
    # generate templated unanswerable explanation
    # unans_explanation = generate_template_nonexistent_select_column_message(schema_modification)
    schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))
    
    removed_cell = copy.deepcopy(schema_modification['removeCell'][0])
    removed_cell.pop("operator", None)

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<column>\n{json.dumps(removed_cell)}\n</column>\n\n<question>\n{line['question']}\n</question>\n\n<sql>\n{line['query']}\n</sql>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user", message_type="litellm")]
    tmp_msgs = [{"role": "system", "content": system_prompt}] + tmp_msgs
    response_obj = litellm_router.completion(messages=tmp_msgs, model="claude-3-sonnet")
    response = response_obj.choices[0].message.content
    # print(user_msg)
    # print(response)
    # print("---" * 20)
    try:
        result = json.loads(extract_string_list_from_xml_tags(response, "result")[0])
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        result = {}

    if len(result) != 2:
        logger.info(f"Invalid results. Shall contain exactly two columns in the replacement. result: {result}")
        logger.info(f"Rmoeve column: {schema_modification}, Related columns: {result}")
        return None

    line['ambiguousUnanswerableConversation'] = {
            "replacementCellCandidates": {
            "raw": response,
            "parsed": result
        }
    }
    line['schemaModification']['addCell'] = result
    # TODO: ideally, we shall call this schemaModification replaceCell as we will delete the cell to remove and replace it with the two replacement cell values

    return line


def replacement_cell_critic_model_for_single_line(
    line, spider_root_dir,
    system_prompt, few_shots,
    litellm_router=None
):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    schema_modification = line.get("schemaModification", {})

    try:
        # we want to get cell values excluding those manually added
        # this is so that we can manually put the manually added cell values at the beginning manually below and ensure they apepar in the schema preview
        # see these two lines:
        #       if num_cells_from_column >= 2:
        #           unique_sampled_cell_values[table][column] = original_cells_from_column[: num_cells_from_column - 2]
        #       unique_sampled_cell_values[table][column] = [item['value'] for item in schema_modification['addCell']] + unique_sampled_cell_values[table][column]
        remove_cell_modification = {
            "removeCell": schema_modification['removeCell']
        }
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=remove_cell_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return {}
    # generate templated unanswerable explanation
    # unans_explanation = generate_template_nonexistent_select_column_message(schema_modification)
    removed_cell = copy.deepcopy(schema_modification['removeCell'][0])
    unique_sampled_cell_values = online_db.get_cell_values(only_unique_value=True, ignore_table_column_casing=False)
    table, column = removed_cell['table'], removed_cell['column']
    grounded_tab_col = online_db.get_grounded_table_column(table=table, column=column)
    grounded_tab, grounded_col = grounded_tab_col['table'], grounded_tab_col['column']
    original_cells_from_column = unique_sampled_cell_values[grounded_tab][grounded_col]
    # MISSING_CELL_VALUE is used to delete a cell from database if the column has constraint that the cell value cannot be None
    # see custom_sql_engine with schema modification
    original_cells_from_column = [cell for cell in original_cells_from_column if cell not in (None, MISSING_CELL_VALUE)]
    # num_cells_from_column = len(original_cells_from_column)
    logger.info(f"Originally sampled cells from column `{column}`: {original_cells_from_column}")
    # put the cells to add at the beginning of the cell value list
    cell_value_to_add = [item['value'] for item in schema_modification['addCell']]
    for original_cell in original_cells_from_column:
        if original_cell not in cell_value_to_add:
            cell_value_to_add.append(original_cell)
    unique_sampled_cell_values[grounded_tab][grounded_col] = cell_value_to_add
    logger.info(f"Sampled cells with newly added cell value from column `{column}`: {unique_sampled_cell_values[grounded_tab][grounded_col]}")

    schema_preview = format_schema_to_markdown(unique_sampled_cell_values)

    removed_cell.pop("operator", None)
    # replacement_cells = line['ambiguousUnanswerableConversation']['replacementCellCandidates']['parsed']
    if "templatedExplanation" in line['ambiguousUnanswerableConversation']:
        templated_explanation = line['ambiguousUnanswerableConversation']['templatedExplanation']
    else:
        unans_explanation = generate_template_nonexistent_select_column_message(schema_modification)
        line['ambiguousUnanswerableConversation']['templatedExplanation'] = unans_explanation
    alternative_sqls = line['ambiguousUnanswerableConversation']['output_response_with_followup_and_sql_parsed']

    # user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<column>\n{json.dumps(removed_cell)}\n</column>\n\n<question>\n{line['question']}\n</question>\n\n<sql>\n{line['query']}\n</sql>\n\n<replacement_cell>\n{json.dumps(replacement_cells, indent=2)}\n</replacement_cell>"
    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<column>\n{json.dumps(removed_cell)}\n</column>\n\n<question>\n{line['question']}\n</question>\n\n<response>\n{templated_explanation}\nThe two possible SQL responses to the question is:\n<sql_list>\n<sql>{alternative_sqls[0]['SQL']}</sql>\n<sql>{alternative_sqls[1]['SQL']}</sql>\n</sql_list>\n<response>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user", message_type="litellm")]
    tmp_msgs = [{"role": "system", "content": system_prompt}] + tmp_msgs
    response_obj = litellm_router.completion(messages=tmp_msgs, model="claude-3-sonnet")
    response = response_obj.choices[0].message.content
    # print(user_msg)
    # print(response)
    # print("---" * 20)
    try:
        result = extract_string_list_from_xml_tags(response, "result")[0].strip()
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        result = None

    if result == "good":
        return line
    else:
        return None


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def generate_followup_sql_for_single_line(line, spider_root_dir, system_prompt, few_shots, litellm_router=None):
    # question = line['question']
    db_id = line['db_id']
    schema_modification = line['schemaModification']
    spider_database_dir = os.path.join(spider_root_dir, "database")

    def _generate_cell_filter_str_list(cell_str):
        cell_str_list = [
            f"'{cell_str}'",
            f'"{cell_str}"',
        ]
        return cell_str_list

    try:
        original_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification={})
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return None

    # fix minor bug when single quote are accidentally added to the key
    if "'replacementCellCandidates'" in line['ambiguousUnanswerableConversation']:
        line['ambiguousUnanswerableConversation']['replacementCellCandidates'] = line['ambiguousUnanswerableConversation']["'replacementCellCandidates'"]
    candidate_cells = line['ambiguousUnanswerableConversation']['replacementCellCandidates']['parsed']

    # generate new SQL by simply replacing the original column with one of the candidate columns
    new_sql_list = []
    for cand_cell in candidate_cells:
        original_sql = copy.deepcopy(line['query'])
        original_cell = schema_modification['removeCell'][0]

        replacement_cell_str = cand_cell['value']
        original_cell_str = original_cell['value']
        original_cell_str_list = _generate_cell_filter_str_list(original_cell_str)
        replacement_cell_str_list = _generate_cell_filter_str_list(replacement_cell_str)

        found_matching_original_cell = False
        for original_str, replacement_str in zip(original_cell_str_list, replacement_cell_str_list):
            if original_str in original_sql:
                new_sql = original_sql.replace(original_str, replacement_str)
                new_sql_list.append(
                    {
                        "SQL": new_sql,
                        "table": cand_cell['table'],
                        "column": cand_cell['column'],
                    }
                )
                found_matching_original_cell = True
                break
            else:
                pass
        if not found_matching_original_cell:
            logger.error(f"None of the column form found in SQL: {original_sql}. Column candidates: {original_cell_str_list}")

    if len(new_sql_list) < 2:
        return None

    response = json.dumps(new_sql_list, indent=2)
    line['ambiguousUnanswerableConversation'].update(
        {
            "output_response_with_followup_and_sql_raw": response,
            "output_response_with_followup_and_sql_parsed": new_sql_list,
        }
    )

    unans_explanation = generate_template_nonexistent_select_column_message(schema_modification)
    line['ambiguousUnanswerableConversation']['templatedExplanation'] = unans_explanation

    return line


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def generate_followup_question_for_single_line(
    line, spider_root_dir, system_prompt, few_shots, litellm_router=None
):
    # generate the conversation template with follow-up question being `FILL-IN-HERE`
    # ask LLM to generate follow-up question for each SQL
    # we shall first filter out the SQL that cannot be executed or resulting empty response
    # then for each follow-up SQL, we generate a follow-up question
    spider_database_dir = os.path.join(spider_root_dir, "database")
    question = line['question']
    schema_modification = line['schemaModification']
    db_id = line['db_id']
    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")

    # generate templated unanswerable explanation
    unans_explanation = generate_template_nonexistent_select_column_message(schema_modification)
    schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))
    followup_sqls = line['ambiguousUnanswerableConversation']['output_response_with_followup_and_sql_parsed']
    valid_sqls = []
    for item in followup_sqls:
        try:
            result = online_db.run_sql(sql=item['SQL'])
            valid_sqls.append(item)
        except Exception as ex:
            logger.error(f"Error running SQL: {ex}")

    followup_question_list = []
    followup_question_raw_list = []
    for item in valid_sqls:
        conversation_list = [
            {
                "USER": question
            },
            {
                "DB EXPERT": unans_explanation,
            },
            {
                "USER": "FILL-IN-YOUR-RESPONSE-HERE"
            },
            {
                "DB EXPERT": item['SQL']
            }
        ]
        user_msg = f"<schema>\n{schema_preview}\n</schema>\n\n<conversation>\n{json.dumps(conversation_list, indent=2)}\n</conversation>"
        tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user", message_type="litellm")]
        tmp_msgs = [{"role": "system", "content": system_prompt}] + tmp_msgs
        response_obj = litellm_router.completion(messages=tmp_msgs, model="claude-3-sonnet")
        response = response_obj.choices[0].message.content
        # print(user_msg)
        # print("---" * 30)
        # print(response)
        try:
            result = extract_string_list_from_xml_tags(response, "result")[0].strip()
        except Exception as ex:
            logger.error(f"Error parsing result: {ex}")
            result = None

        followup_question_raw_list.append(
            {
                "USER": response,
                "DB EXPERT": item['SQL'],
            }
        )

        followup_question_list.append(
            {
                "USER": result,
                "DB EXPERT": item['SQL'],
            }
        )

    # if there is no followup question, return None
    if len(followup_question_raw_list) == 0:
        return None

    # set the followup and response to fixed template for reranking
    conversation_list[-2]['USER'] = "FILL-IN-YOUR-RESPONSE-HERE"
    conversation_list[-1]['DB EXPERT'] = "FILL-IN-YOUR-SQL-RESPONSE-HERE"

    line['ambiguousUnanswerableConversation'].update(
        {
            "input_conversation_with_placeholder_to_fill": conversation_list,
            "output_response_with_followup_and_sql_raw": followup_question_raw_list,
            "output_response_with_followup_and_sql_parsed": followup_question_list,
        }
    )
    logger.info(f"Followup question generated for {conversation_list}:\n {json.dumps(followup_question_list, indent=2)}")
    return line


def generate_template_nonexistent_select_column_message(schema_modification):
    original_cell = schema_modification['removeCell'][0]['value']
    cell1, cell2 = schema_modification['addCell'][0]['value'], schema_modification['addCell'][1]['value']
    # table = schema_modification['removeCell'][0]['table']
    column = schema_modification['removeCell'][0]['column']
    template = f"The question is ambiguous. The question is asking to filter results by '{original_cell}' from column '{column}'. However, the schema does not contain that cell value '{original_cell}'. Instead, the schema contains cell value '{cell1}' and '{cell2}'. It is not clear which cell value I shall use to answer the question."
    return template


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def select_most_natural_followup_for_split(
    line,
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict],
    litellm_router=None,
):

    spider_database_dir = os.path.join(spider_root_dir, "database")
    schema_modification = line['schemaModification']

    db_id = line['db_id']
    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        # online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
        return None

    templated_conversation = line['ambiguousUnanswerableConversation']['input_conversation_with_placeholder_to_fill']
    candidate_follow_ups = line['ambiguousUnanswerableConversation'].get('output_response_with_followup_and_sql_parsed')
    if not candidate_follow_ups:
        candidate_followup_raw = line['ambiguousUnanswerableConversation']['output_response_with_followup_and_sql_raw']
        # parse the results
        candidate_follow_ups = extract_string_list_from_xml_tags(candidate_followup_raw, "result")

    # if we only have one single valid follow-up SQL and response somehow, no need to select, directly return
    if len(candidate_follow_ups) == 1:
        result = candidate_follow_ups
        response = json.dumps(result, indent=2)
    elif len(candidate_follow_ups) == 0:
        return None
    else:
        schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))
        user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<conversation>\n\n{json.dumps(templated_conversation, indent=2)}\n\n</conversation>\n\n<follow-up>\n{json.dumps(candidate_follow_ups, indent=2)}\n</follow-up>"
        tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user", message_type="litellm")]
        tmp_msgs = [{"role": "system", "content": system_prompt}] + tmp_msgs
        response_obj = litellm_router.completion(messages=tmp_msgs, model="claude-3-sonnet")
        response = response_obj.choices[0].message.content
        # print(user_msg)
        # print("---" * 30)
        # print(response)
        try:
            result = json.loads(extract_string_list_from_xml_tags(response, "result")[0])
        except Exception as ex:
            logger.error(f"Error parsing result: {ex}")
            result = {}
    result_to_update = {
        "ranked_response_with_followup_sql_raw": response,
        "ranked_response_with_followup_sql_parsed": result,
    }
    line['ambiguousUnanswerableConversation'].update(result_to_update)
    return line


def remove_invalid_followup_based_on_sql_execution(line, spider_root_dir: str):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    schema_modification = line['schemaModification']

    db_id = line['db_id']
    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return None

    conversation_data = line['ambiguousUnanswerableConversation']
    ranked_sql_parsed = conversation_data.get("ranked_response_with_followup_sql_parsed", None)
    if not ranked_sql_parsed:
        ranked_sql_parsed = extract_string_list_from_xml_tags(
            text=conversation_data["ranked_response_with_followup_sql_raw"], tag_name="result"
        )
    valid_followup_sql = []
    for followup_sql in ranked_sql_parsed:
        sql = followup_sql['DB EXPERT']
        if online_db.is_sql_executable(sql):
            execution_result = online_db.run_sql(sql)
            followup_sql['SQL_EXECUTION_RESULTS'] = execution_result
            valid_followup_sql.append(followup_sql)
        else:
            logger.warning(f"Invalid followup sql: {sql}")
    line['ambiguousUnanswerableConversation']['valid_ranked_response_with_followup_sql_parsed'] = valid_followup_sql
    return line


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def rephrase_the_templated_explanation_single_line(
    line,
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict],
    litellm_router=None,
):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    schema_modification = line['schemaModification']
    db_id = line['db_id']

    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        # online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)

    removed_cell = copy.deepcopy(schema_modification['removeCell'][0])
    unique_sampled_cell_values = online_db.get_cell_values(only_unique_value=True, ignore_table_column_casing=False)
    table, column = removed_cell['table'], removed_cell['column']
    grounded_tab_col = online_db.get_grounded_table_column(table=table, column=column)
    grounded_tab, grounded_col = grounded_tab_col['table'], grounded_tab_col['column']
    original_cells_from_column = unique_sampled_cell_values[grounded_tab][grounded_col]
    # MISSING_CELL_VALUE is used to delete a cell from database if the column has constraint that the cell value cannot be None
    # see custom_sql_engine with schema modification
    original_cells_from_column = [cell for cell in original_cells_from_column if cell not in (None, MISSING_CELL_VALUE)]
    # num_cells_from_column = len(original_cells_from_column)
    logger.info(f"Originally sampled cells from column `{column}`: {original_cells_from_column}")
    # put the cells to add at the beginning of the cell value list
    cell_value_to_add = [item['value'] for item in schema_modification['addCell']]
    for original_cell in original_cells_from_column:
        if original_cell not in cell_value_to_add:
            cell_value_to_add.append(original_cell)
    unique_sampled_cell_values[grounded_tab][grounded_col] = cell_value_to_add
    logger.info(f"Sampled cells with newly added cell value from column `{column}`: {unique_sampled_cell_values[grounded_tab][grounded_col]}")

    schema_preview = format_schema_to_markdown(unique_sampled_cell_values)

    conversation_with_template = line['ambiguousUnanswerableConversation']['input_conversation_with_placeholder_to_fill']
    conversation_with_followup = copy.deepcopy(conversation_with_template)
    for followup_sql in line['ambiguousUnanswerableConversation']['valid_ranked_response_with_followup_sql_parsed']:
        conversation_with_followup[-2] = {
            "USER": followup_sql['USER'],
        }
        conversation_with_followup[-1] = {
            "DB EXPERT": followup_sql['DB EXPERT'],
        }
        break

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<conversation>\n\n{json.dumps(conversation_with_followup, indent=2)}\n\n</conversation>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user", message_type="litellm")]
    tmp_msgs = [{"role": "system", "content": system_prompt}] + tmp_msgs
    response_obj = litellm_router.completion(messages=tmp_msgs, model="claude-3-sonnet")
    response = response_obj.choices[0].message.content
    # print(user_msg)
    # print("---" * 30)
    # print(response)
    try:
        result = extract_string_list_from_xml_tags(response, "result")[0].strip()
        final_conversation_with_followup = copy.deepcopy(conversation_with_followup)
        final_conversation_with_followup[1]['DB EXPERT'] = result
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        final_conversation_with_followup = {}
        result = {}
    
    result_to_update = {
        "rephrased_explanation_selected_followup_sql_raw": response,
        "rephrased_explanation_selected_followup_sql_parsed": result,
        "rephrased_explanation_selected_followup_sql_complete_conversation": final_conversation_with_followup 
    }
    line['ambiguousUnanswerableConversation'].update(result_to_update)
    return line


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def add_semantic_equivalent_columns_to_schema_modification_and_check_removed_col_in_sql(line, spider_root_dir: str):
    # spider_database_dir = os.path.join(spider_root_dir, "database")
    schema_modification = line['schemaModification']
    llm_identified_columns = schema_modification.get("LLM_Based_Alternative_Columns_To_Remove")
    semantic_cols = llm_identified_columns.get("parsed")
    if not semantic_cols:
        semantic_cols_raw = llm_identified_columns.get("raw")
        semantic_cols = json.loads(extract_string_list_from_xml_tags(semantic_cols_raw, "result")[0])
    schema_modification['removeColumnSemanticallyRelated'] = semantic_cols

    columns_in_sql = get_all_table_column_info(sql_query=line['query'])

    # add the semantic equivalent columns to the schema modification for modification
    line['schemaModification']['removeColumnSemanticallyRelated'] = semantic_cols
    # in_memory_db = DbWithModification(db_id_name=line['db_id'], database_main_dir=spider_database_dir, schema_modification=schema_modification)
    # check whether other columns rather than the primary column removed are mentioned in the SQL
    # if other columns in the SQL are removed as a lexically ore semantically related columns, skip this line
    removed_related_columns = schema_modification['removeColumnSemanticallyRelated'] + schema_modification['removeColumnLexicallyRelated']
    removed_extra_column_in_sql = False
    if removed_related_columns:
        for rm_tab_col in removed_related_columns:
            for sql_tab_col in columns_in_sql:
                # if rm_tab_col['table'].lower() == sql_tab_col['table'].lower() and rm_tab_col['column'].lower() == sql_tab_col['column'].lower():
                #     removed_extra_column_in_sql = True
                #     break
                try:
                    if rm_tab_col['table'].lower() == sql_tab_col['table'].lower() and rm_tab_col['column'].lower() == sql_tab_col['column'].lower():
                        removed_extra_column_in_sql = True
                        break
                except Exception as ex:
                    logger.error(f"Error `{ex}` in determining whether additional columns used in the SQL are removed in schema modification.\nRemoved related columns: {rm_tab_col}\nColumns used in the SQL: {sql_tab_col}\nLine data for debugging: {json.dumps(line, indent=2)}")

    if removed_extra_column_in_sql:
        return None

    return line


def process_single_line(
    line: Dict,
    spider_root_dir,
    litellm_router=None,
    replacement_cell_system_prompt=None,
    replacement_cell_fewshots=None,
    critic_model_system_prompt=None,
    critic_model_fewshots=None,
    followup_question_system_prompt=None,
    followup_question_few_shots=None,
    select_natural_system_prompt=None,
    select_natural_few_shots=None,
    rephrase_templated_explanation_system_prompt=None,
    rephrase_templated_explanation_few_shots=None,

):
    """
    Process a single line to generate ambiguous values within column examples.

    This function is designed to work with multiprocessing. ALL exceptions must be
    converted to standard pickleable exceptions to avoid serialization issues.
    """
    try:
        return _process_single_line_impl(
            line=line,
            spider_root_dir=spider_root_dir,
            litellm_router=litellm_router,
            replacement_cell_system_prompt=replacement_cell_system_prompt,
            replacement_cell_fewshots=replacement_cell_fewshots,
            critic_model_system_prompt=critic_model_system_prompt,
            critic_model_fewshots=critic_model_fewshots,
            followup_question_system_prompt=followup_question_system_prompt,
            followup_question_few_shots=followup_question_few_shots,
            select_natural_system_prompt=select_natural_system_prompt,
            select_natural_few_shots=select_natural_few_shots,
            rephrase_templated_explanation_system_prompt=rephrase_templated_explanation_system_prompt,
            rephrase_templated_explanation_few_shots=rephrase_templated_explanation_few_shots,
        )
    except Exception as ex:
        # Convert ALL exceptions (including litellm exceptions) to pickleable standard exceptions
        # Use 'from None' to break exception chain and avoid unpickleable parent exceptions
        error_msg = f"{type(ex).__name__}: {str(ex)}"
        logger.error(f"Error processing line in worker: {error_msg}")
        raise Exception(error_msg) from None


def _process_single_line_impl(
    line: Dict,
    spider_root_dir,
    litellm_router=None,
    replacement_cell_system_prompt=None,
    replacement_cell_fewshots=None,
    critic_model_system_prompt=None,
    critic_model_fewshots=None,
    followup_question_system_prompt=None,
    followup_question_few_shots=None,
    select_natural_system_prompt=None,
    select_natural_few_shots=None,
    rephrase_templated_explanation_system_prompt=None,
    rephrase_templated_explanation_few_shots=None,

):
    """Internal implementation of process_single_line."""
    # Create a new router in each worker process to avoid serialization issues
    if litellm_router is None:
        litellm_router = get_litellm_router()

    line_with_text_filter = get_sql_cell_filter(line=line)
    if not line_with_text_filter:
        logger.info(f"No text filter found in line: {line['query']}")
        return None

    # identify additional cell values to remove that are similar to that needs to be deleted
    line_with_additional_cell_values_to_remove = identify_additional_cell_values_to_remove(
        line=line_with_text_filter,
        spider_root_dir=spider_root_dir,
        system_prompt="",
        few_shots=[],
        litellm_router=litellm_router,
    )
    if not line_with_additional_cell_values_to_remove:
        return None

    # TODO: remove semantically related cell values
    # generating replacement cell values and modify SQL accordinly
    line_with_replacement_cells = generate_replacement_cell_values_for_single_line(
        line=line_with_additional_cell_values_to_remove,
        spider_root_dir=spider_root_dir,
        system_prompt=replacement_cell_system_prompt,
        few_shots=replacement_cell_fewshots, 
        litellm_router=litellm_router,
    )
    if not line_with_replacement_cells:
        return None
    # return line_with_replacement_cells

    # generate follow-up SQL based on columns
    line_with_follow_up_sql = generate_followup_sql_for_single_line(
        line=line_with_replacement_cells,
        spider_root_dir=spider_root_dir,
        system_prompt="",
        few_shots=[], 
        litellm_router=litellm_router,
    )
    if not line_with_follow_up_sql:
        return None

    # # critic model: check whether the two replacement SQL are equal explanations of the original question
    line_with_critic_replacement_cells = replacement_cell_critic_model_for_single_line(
        line=line_with_follow_up_sql,
        spider_root_dir=spider_root_dir,
        system_prompt=critic_model_system_prompt,
        few_shots=critic_model_fewshots,
        litellm_router=litellm_router,
    )
    if not line_with_critic_replacement_cells:
        return None

    # generate follow-up question via reverse generation
    line_with_followup_question_sql = generate_followup_question_for_single_line(
        line=line_with_critic_replacement_cells,
        spider_root_dir=spider_root_dir,
        system_prompt=followup_question_system_prompt,
        few_shots=followup_question_few_shots,
        litellm_router=litellm_router,
    )
    if not line_with_followup_question_sql:
        return None

    # select most natural follow-up sql and question
    line_with_ranked_followup = select_most_natural_followup_for_split(
        line=line_with_followup_question_sql,
        spider_root_dir=spider_root_dir,
        system_prompt=select_natural_system_prompt,
        few_shots=select_natural_few_shots,
        litellm_router=litellm_router,
    )
    if line_with_ranked_followup is None:
        return None

#     # TODO: remove the invalid sql from candidates
    try:
        line_with_ranked_followup = remove_invalid_followup_based_on_sql_execution(line=line_with_ranked_followup, spider_root_dir=spider_root_dir)
    except Exception as ex:
        logger.error(f"Exceptions in removing invalid SQLs: {ex}")
        logger.error(f"Current line content for debugging:\n{json.dumps(line_with_ranked_followup, indent=2)}")
        return None

    # rephrase the templated explanation
    line_with_rephrased_explanation = rephrase_the_templated_explanation_single_line(
        line=line_with_ranked_followup,
        spider_root_dir=spider_root_dir,
        system_prompt=rephrase_templated_explanation_system_prompt,
        few_shots=rephrase_templated_explanation_few_shots,
        litellm_router=litellm_router,
    )
    if line_with_rephrased_explanation is None:
        return None
    return line_with_rephrased_explanation



def process_lines_of_interest(
    lines,
    spider_root_dir: str,
    num_workers: int = 7,
):

    # generating replacement cell values
    replacement_cell_fewshots_fn = "fewshots_examples_generating_replacement_cell_values.ipynb"
    replacement_cell_fewshots_path = os.path.join(CURRENT_DIR, replacement_cell_fewshots_fn)
    replacement_cell_fewshots, replacement_cell_system_prompt = add_fewshots_from_path(path_str=replacement_cell_fewshots_path, extension=".ipynb")
    replacement_cell_fewshots = convert_claude_msg_list_to_litellm_msg_list(replacement_cell_fewshots)

    # having a critic model to ensure that the generated cell values are all valid
    critic_model_fewshots_fn = "fewshots_replacement_cell_critic_based_on_sql.ipynb"
    critic_model_fewshots_path = os.path.join(CURRENT_DIR, critic_model_fewshots_fn)
    critic_model_fewshots, critic_model_system_prompt = add_fewshots_from_path(path_str=critic_model_fewshots_path, extension=".ipynb")
    critic_model_fewshots = convert_claude_msg_list_to_litellm_msg_list(critic_model_fewshots)

    # follow-up question generation
    followup_question_few_shots_filename = "fewshots_generating_followup_question_from_sql.ipynb"
    followup_question_few_shots_path = os.path.join(CURRENT_DIR, followup_question_few_shots_filename)
    followup_question_few_shots, followup_question_system_prompt = add_fewshots_from_path(path_str=followup_question_few_shots_path, extension=".ipynb")
    followup_question_few_shots = convert_claude_msg_list_to_litellm_msg_list(followup_question_few_shots)

    # select most natural followup and SQL
    select_natural_few_shots_filename = "fewshots_examples_selecting_most_natural_follow_up_refined_system_prompt.ipynb"
    select_natural_few_shots_path = os.path.join(CURRENT_DIR, select_natural_few_shots_filename)
    select_natural_few_shots, select_natural_system_prompt = add_fewshots_from_path(path_str=select_natural_few_shots_path, extension=".ipynb")
    select_natural_few_shots = convert_claude_msg_list_to_litellm_msg_list(select_natural_few_shots)

    # rephrase the templated explanation
    rephrase_templated_explanation_few_shots_filename = "fewshots_examples_for_rephrasing_templated_unanswerable_response.ipynb"
    rephrase_templated_explanation_few_shots_path = os.path.join(CURRENT_DIR, rephrase_templated_explanation_few_shots_filename)
    rephrase_templated_explanation_few_shots, rephrase_templated_explanation_system_prompt = add_fewshots_from_path(path_str=rephrase_templated_explanation_few_shots_path, extension=".ipynb")
    rephrase_templated_explanation_few_shots = convert_claude_msg_list_to_litellm_msg_list(rephrase_templated_explanation_few_shots)


    nlines = []
    # Note: Router is created in each worker process, not here
    # Sequential processing (uncomment if needed):
    # router = get_litellm_router()
    # for line in tqdm(lines):
    #     line_with_rephrased_explanation = process_single_line(line=line, spider_root_dir=spider_root_dir, litellm_router=router)
    #     if line_with_rephrased_explanation:
    #         nlines.append(line_with_rephrased_explanation)

    # Use configurable number of workers to avoid overwhelming API rate limits
    with multiprocessing.Pool(processes=num_workers) as pool:
        nlines = list(tqdm(pool.imap(
            partial(
                process_single_line, spider_root_dir=spider_root_dir,
                # Don't pass litellm_router - each worker will create its own to avoid serialization issues
                replacement_cell_system_prompt=replacement_cell_system_prompt,
                replacement_cell_fewshots=replacement_cell_fewshots,
                critic_model_system_prompt=critic_model_system_prompt,
                critic_model_fewshots=critic_model_fewshots,
                followup_question_system_prompt=followup_question_system_prompt,
                followup_question_few_shots=followup_question_few_shots,
                select_natural_system_prompt=select_natural_system_prompt,
                select_natural_few_shots=select_natural_few_shots,
                rephrase_templated_explanation_system_prompt=rephrase_templated_explanation_system_prompt,
                rephrase_templated_explanation_few_shots=rephrase_templated_explanation_few_shots,
            ),
            lines
        ), total=len(lines)))


    return nlines


def main(
    spider_data_root_dir: str = typer.Option(
        '../../spider/dataset', help='The path to the Spider dataset'
    ),
    output_dir: str = typer.Option(
        '.vscode/output_2024_05', help='The path to the output directory'
    ),
    split: str = typer.Option('dev', help='The split to use: dev or train or all'),
    n2sample: int = typer.Option(0, help='Sample the questions'),
    num_workers: int = typer.Option(1, help='Number of parallel workers for processing'), # if main shell script run generation for 8 categories in parallel, this shall be a small number to avoid throttling
):

    if split == "all":
        dev_all = load_spider_dev_train_data(spider_data_root_dir=spider_data_root_dir, output_dir=output_dir, split="dev")
        train_all = load_spider_dev_train_data(spider_data_root_dir=spider_data_root_dir, output_dir=output_dir, split="train")
        split_all = dev_all + train_all
    else:
        split_all = load_spider_dev_train_data(spider_data_root_dir=spider_data_root_dir, output_dir=output_dir, split=split)

    # # # extract questions with join and include metadata: ambiguousUnanswerableCategory & schemaModification
    # lines_of_interest = get_questions_with_select_clause(spider_data=split_all, spider_root_dir=spider_data_root_dir)

    if n2sample:
        split_all = sample_questions_by_database(lines=split_all, n_question_per_db=n2sample)

    nlines = process_lines_of_interest(
        lines=split_all,
        spider_root_dir=spider_data_root_dir,
        num_workers=num_workers,
    )

    nlines = [line for line in nlines if line]
    output_jsonl_fp = os.path.join(output_dir, split, AMB_UNANS_CATEGORY) + ".jsonl"
    logger.info(f"Output dir: {output_jsonl_fp}")
    write_jsonl(nlines, output_jsonl_fp)
    

if __name__ == '__main__':
    # TODO: fixed the fewshots example with new templated ambiguosu explanation
    # TODO: add a filtering step to remove un-natural conversations
    typer.run(main)
