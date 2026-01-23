# Standard library imports
import json 
import os
import random
import copy
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# Third-party imports
import typer
from loguru import logger
from tqdm import tqdm
from typing import Any, Optional, Tuple

# Local imports
from simple_cache import cache_results
from custom_sql_engine import DbWithModification
from utils import (
    set_random_seed,
    format_schema_to_markdown,
    create_simple_message,
    write_jsonl,
    get_select_table_column_info,
    get_lexically_similar_columns_from_schema,
    load_spider_dev_train_data,
    get_all_table_column_info,
    sample_questions_by_database,
)
from helpers import (
    add_fewshots_from_path,
    extract_string_list_from_xml_tags,
)
from litellm_helpers import (
    get_litellm_router,
    convert_claude_msg_list_to_litellm_msg_list,
    router_completion_with_ratelimit_retry,
)


AMB_UNANS_CATEGORY = "Nonexistent_SELECT_Column"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "__cache__")  # use current dir as a cache dire
os.makedirs(CACHE_DIR, exist_ok=True)
IGNORE_CACHE = True
CURRENT_DIR = os.path.dirname(__file__)

set_random_seed(7)


@cache_results(CACHE_DIR, ignore_cache=IGNORE_CACHE)
def identify_select_column_for_removal(line: Dict[str, Any], spider_root_dir: str) -> Optional[Dict[str, Any]]:
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    parsed_select_clause = get_select_table_column_info(line['query'])[0]
    # if len(parsed_select_clause) > 1:
    #     logger.debug(f"More than one select clause: {line['query']}")
    if parsed_select_clause:
        # get the unique select clause 
        # NOTE: we can add all unique SELECT clause as one unanswerable case, to balance, here, we randomly select one for unanswerable data generation
        # unique_select_clause_select = get_unique_select_clause(parsed_select_clause)
        # for single_select_clause in unique_select_clause_select:
        #     # single_select_clause = random.choice(parsed_select_clause)
        #     line['ambiguousUnanswerableCategory'] = AMB_UNANS_CATEGORY
        #     line['schemaModification'] = {
        #         "removeColumn": [
        #             {
        #                 "table": single_select_clause['table'], "column": single_select_clause['column']
        #             }
        #         ]
        #     }
        #     questions_of_interest.append(line)
        #     total_line_of_interest += 1

        # remove invalid Count SELECT columns they often can be determined by other alternative columns
        # e.g., number of employees in from Psychology. department table can have both employee_name & employee_id column used for count
        valid_select_clause_list = []
        sql_lower = line['query'].lower()
        for select_clause in parsed_select_clause:
            col_name = select_clause['column']
            count_phrases = [
                f"count({col_name})",
                f"count('{col_name}')",
                f'count("{col_name}")',
            ]
            if all([phrase not in sql_lower for phrase in count_phrases]):
                valid_select_clause_list.append(select_clause)

        if len(valid_select_clause_list) == 0:
            return None

        # NOTE: we randomly choose a SELECT clause for unanswerable data generation
        single_select_clause = random.choice(valid_select_clause_list)
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=None)
        tab_col_cells = online_db.get_cell_values(only_unique_value=True)
        lexically_related_columns = get_lexically_similar_columns_from_schema(single_select_clause, tab_col_cells=tab_col_cells)
        line['ambiguousUnanswerableCategory'] = AMB_UNANS_CATEGORY
        line['schemaModification'] = {
            "removeColumn": [
                {
                    "table": single_select_clause['table'], "column": single_select_clause['column']
                }
            ],
            "removeColumnLexicallyRelated": lexically_related_columns,
        }
        return line
    else:
        if "select" in line['query'].lower() and "*" not in line['query']:
            logger.debug(f"No select clause: {line['query']}")
        # logger.debug(f"No where clause: {data['query']}")
        return None


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def identify_alternative_columns_for_deletion_single_line(
    line: Dict[str, Any],
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    schema_modification = line.get("schemaModification", {})

    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return {}
    # generate templated unanswerable explanation
    # unans_explanation = generate_template_nonexistent_select_column_message(schema_modification)
    schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<column>\n{json.dumps(schema_modification['removeColumn'][0])}\n</column>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
    try:
        result = json.loads(extract_string_list_from_xml_tags(response, "result")[0])
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        result = {}

    if len(result) > 0:
        logger.info(f"Rmoeve column: {schema_modification}, Related columns: {result}")
    schema_modification_update = {
        "LLM_Based_Alternative_Columns_To_Remove": {
            "raw": response,
            "parsed": result
        }
    }
    line['schemaModification'].update(schema_modification_update)
    return line


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def generate_alternative_columns_to_ask_for_single_line(
    line: Dict[str, Any],
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    schema_modification = line.get("schemaModification", {})

    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return {}
    # generate templated unanswerable explanation
    # unans_explanation = generate_template_nonexistent_select_column_message(schema_modification)
    schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<column>\n{json.dumps(schema_modification['removeColumn'][0])}\n</column>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
    try:
        result = json.loads(extract_string_list_from_xml_tags(response, "result")[0])
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        result = []

    if len(result) > 0:
        logger.info(f"Rmoeve column: {schema_modification}, Related columns: {result}")

    # dedupe the result based on uniqueness of column names
    deduped_result = []
    unique_columns = []
    for tab_col in result:
        col = tab_col['column']
        if col.strip().lower() not in unique_columns:
            unique_columns.append(col.strip().lower())
            deduped_result.append(tab_col)
        else:
            logger.info(f"Candidate alternative column to ask removed due to duplication. Deduped column: {col}. Candidate column pool: {unique_columns}")

    line['ambiguousUnanswerableConversation'] = {
            "replacementColumnCandidates": {
            "raw": response,
            "parsed_without_deduplication": result,
            "parsed": deduped_result,
        }
    }
    return line


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def generate_followup_sql_for_single_line(
    line: Dict[str, Any],
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    db_id = line['db_id']
    schema_modification = line['schemaModification']
    spider_database_dir = os.path.join(spider_root_dir, "database")
    
    try:
        original_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification={})
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return None

    # generate templated unanswerable explanation
    tabl_col_sample_cells = online_db.get_cell_values(only_unique_value=True)
    schema_preview = format_schema_to_markdown(tabl_col_sample_cells)
    all_tab_col_in_sql = get_all_table_column_info(line['query'])
    all_tab_col_str_list = [
        f"{tab_col['table']}_{tab_col['column']}".lower()
        for tab_col in all_tab_col_in_sql
    ]

    # fix minor bug when single quote are accidentally added to the key
    if "'replacementColumnCandidates'" in line['ambiguousUnanswerableConversation']:
        line['ambiguousUnanswerableConversation']['replacementColumnCandidates'] = line['ambiguousUnanswerableConversation']["'replacementColumnCandidates'"]
    candidate_columns = line['ambiguousUnanswerableConversation']['replacementColumnCandidates']['parsed']

    # replace the removed column in SQL with new column if they are of the same physical type
    # generate the replacement SQL
    original_tab_col_type_mapping = original_db.get_table_column_type_mapping(lower_case=True)
    target_tab, target_col = schema_modification['removeColumn'][0]['table'], schema_modification['removeColumn'][0]['column']
    target_col_type = original_tab_col_type_mapping.get(target_tab.lower(), {}).get(target_col.lower(), "UNKNOWN")
    table_col_type_map = online_db.get_table_column_type_mapping(lower_case=True)
    valid_columns = []
    for cand_tab_col in candidate_columns:
        cand_tab, cand_col = cand_tab_col['table'], cand_tab_col['column']
        cand_tab_col_str_lower = f"{cand_tab}_{cand_col}".lower()
        cand_col_type = table_col_type_map.get(cand_tab.lower(), {}).get(cand_col.lower(), '')
        if cand_tab_col_str_lower not in all_tab_col_str_list and cand_col.lower() != target_col.lower():
            valid_columns.append(cand_tab_col)

    # for column from the same table, we can do simple replacement, for column from other table, we will need to leverage LLM for that
    # generate the candidate SQLs based on the valid columns
    replacement_column_str = json.dumps(valid_columns, indent=2)
    original_sql = line['query']
    original_column_str = json.dumps(schema_modification['removeColumn'][0])
    user_msg = f"<schema>\n{schema_preview}\n</schema>\n\n<original_sql>\n{original_sql}\n</original_sql>\n\n<original_column>\n{original_column_str}\n</original_column>\n\n<replacement_columns>\n{replacement_column_str}\n</replacement_columns>"

    # print(user_msg)
    # print("---" * 30)
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
    try:
        result = json.loads(extract_string_list_from_xml_tags(response, "result")[0])
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        result = {}
    line['ambiguousUnanswerableConversation'].update(
        {
            "output_response_with_followup_and_sql_raw": response,
            "output_response_with_followup_and_sql_parsed": result,
        }
    )
    return line


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def generate_followup_question_for_single_line(
    line: Dict[str, Any],
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
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

        # print(user_msg)
        # print("---" * 30)
        tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
        response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
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


def generate_template_nonexistent_select_column_message(schema_modification: Dict[str, Any]) -> str:
    removed_column_info = schema_modification['removeColumn']
    table = removed_column_info[0]['table']
    column = removed_column_info[0]['column']

    return f"To answer the question, we need a column like `{column}` from table `{table}`. However, such column does not exist. Can you ask a different question?"


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def select_most_natural_followup_for_split(
    line: Dict[str, Any],
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
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
    schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))
    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<conversation>\n\n{json.dumps(templated_conversation, indent=2)}\n\n</conversation>\n\n<follow-up>\n{json.dumps(candidate_follow_ups, indent=2)}\n</follow-up>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
    print(user_msg)
    print("---" * 30)
    print(response)
    print("===" * 30)
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


def remove_invalid_followup_based_on_sql_execution(line: Dict[str, Any], spider_root_dir: str) -> Optional[Dict[str, Any]]:
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
    line: Dict[str, Any],
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    spider_database_dir = os.path.join(spider_root_dir, "database")
    schema_modification = line['schemaModification']
    db_id = line['db_id']

    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        # online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)

    schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))
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
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
    # print(user_msg)
    # print("---" * 30)
    # print(response)
    # print("===" * 30)
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
def add_semantic_equivalent_columns_to_schema_modification_and_check_removed_col_in_sql(
    line: Dict[str, Any],
    spider_root_dir: str
) -> Optional[Dict[str, Any]]:
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
                try:
                    if rm_tab_col['table'].lower() == sql_tab_col['table'].lower() and rm_tab_col['column'].lower() == sql_tab_col['column'].lower():
                        removed_extra_column_in_sql = True
                        break
                except Exception as ex:
                    logger.error(f"Error `{ex}` in determining whether additional columns used in the SQL are removed in schema modification.\nRemoved related columns: {rm_tab_col}\nColumns used in the SQL: {sql_tab_col}\nLine data for debugging: {json.dumps(line, indent=2)}")

    if removed_extra_column_in_sql:
        return None

    return line


def load_few_shots(filename: str) -> Tuple[List[Dict[str, Any]], str]:
    few_shots_path = os.path.join(CURRENT_DIR, filename)
    return add_fewshots_from_path(path_str=few_shots_path, extension=".ipynb")

def process_single_line(
    line: Dict[str, Any],
    spider_root_dir: str,
    few_shots: Dict[str, Tuple[List[Dict[str, Any]], str]],
) -> Optional[Dict[str, Any]]:
    try:
        logger.info(f"Processing line with db_id: {line.get('db_id')}")

        line_with_select_clause = identify_select_column_for_removal(line=line, spider_root_dir=spider_root_dir)
        if not line_with_select_clause:
            logger.warning(f"No select clause identified for db_id: {line.get('db_id')}")
            return None

        line_with_semantic_similar_columns = identify_alternative_columns_for_deletion_single_line(
            line=line_with_select_clause,
            spider_root_dir=spider_root_dir,
            system_prompt=few_shots['alternative_columns_to_delete'][1],
            few_shots=few_shots['alternative_columns_to_delete'][0],
        )
        if not line_with_semantic_similar_columns:
            logger.warning(f"No semantically similar columns identified for db_id: {line.get('db_id')}")
            return None

        line_with_semantic_similar_columns = add_semantic_equivalent_columns_to_schema_modification_and_check_removed_col_in_sql(
            line=line_with_semantic_similar_columns,
            spider_root_dir=spider_root_dir,
        )
        if not line_with_semantic_similar_columns:
            logger.warning(f"Failed to add semantic equivalent columns for db_id: {line.get('db_id')}")
            return None

        line_with_alternative_columns_to_ask = generate_alternative_columns_to_ask_for_single_line(
            line=line_with_semantic_similar_columns,
            spider_root_dir=spider_root_dir,
            system_prompt=few_shots['alternative_columns_to_ask'][1],
            few_shots=few_shots['alternative_columns_to_ask'][0],
        )
        if not line_with_alternative_columns_to_ask:
            logger.warning(f"Failed to generate alternative columns to ask for db_id: {line.get('db_id')}")
            return None

        line_with_follow_up_sql = generate_followup_sql_for_single_line(
            line=line_with_alternative_columns_to_ask,
            spider_root_dir=spider_root_dir,
            system_prompt=few_shots['followup_sql'][1],
            few_shots=few_shots['followup_sql'][0],
        )
        if not line_with_follow_up_sql:
            logger.warning(f"Failed to generate follow-up SQL for db_id: {line.get('db_id')}")
            return None

        line_with_followup_question_sql = generate_followup_question_for_single_line(
            line=line_with_follow_up_sql,
            spider_root_dir=spider_root_dir,
            system_prompt=few_shots['followup_question'][1],
            few_shots=few_shots['followup_question'][0],
        )
        if not line_with_followup_question_sql:
            logger.warning(f"Failed to generate follow-up question SQL for db_id: {line.get('db_id')}")
            return None

        line_with_ranked_followup = select_most_natural_followup_for_split(
            line=line_with_followup_question_sql,
            spider_root_dir=spider_root_dir,
            system_prompt=few_shots['select_natural'][1],
            few_shots=few_shots['select_natural'][0],
        )
        if not line_with_ranked_followup:
            logger.warning(f"Failed to select most natural follow-up for db_id: {line.get('db_id')}")
            return None

        line_with_ranked_followup = remove_invalid_followup_based_on_sql_execution(line=line_with_ranked_followup, spider_root_dir=spider_root_dir)
        if not line_with_ranked_followup:
            logger.warning(f"Failed to remove invalid follow-ups for db_id: {line.get('db_id')}")
            return None

        line_with_rephrased_explanation = rephrase_the_templated_explanation_single_line(
            line=line_with_ranked_followup,
            spider_root_dir=spider_root_dir,
            system_prompt=few_shots['rephrase_templated_explanation'][1],
            few_shots=few_shots['rephrase_templated_explanation'][0],
        )
        if not line_with_rephrased_explanation:
            logger.warning(f"Failed to rephrase templated explanation for db_id: {line.get('db_id')}")
            return None

        logger.info(f"Successfully processed line with db_id: {line.get('db_id')}")
        return line_with_rephrased_explanation

    except Exception as e:
        logger.error(f"Error processing line with db_id: {line.get('db_id')}. Error: {str(e)}")
        return None

def process_lines_of_interest(
    lines: List[Dict[str, Any]],
    spider_root_dir: str,
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    few_shots = {
        'alternative_columns_to_delete': load_few_shots("fewshots_examples_identifying_alternative_columns_to_delete.ipynb"),
        'alternative_columns_to_ask': load_few_shots("fewshots_alternative_columns_to_ask.ipynb"),
        'followup_sql': load_few_shots("fewshots_generating_sql_from_alternative_columns_to_ask.ipynb"),
        'followup_question': load_few_shots("fewshots_generating_followup_question_from_sql.ipynb"),
        'select_natural': load_few_shots("fewshots_examples_selecting_most_natural_follow_up_refined_system_prompt.ipynb"),
        'rephrase_templated_explanation': load_few_shots("fewshots_examples_for_rephrasing_templated_unanswerable_response.ipynb"),
    }

    result_queue = Queue()

    def process_and_enqueue(line):
        processed_line = process_single_line(line, spider_root_dir, few_shots)
        if processed_line:
            result_queue.put(processed_line)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_and_enqueue, line) for line in lines]
        
        for _ in tqdm(as_completed(futures), total=len(lines), desc="Processing lines"):
            pass

    nlines = []
    while not result_queue.empty():
        nlines.append(result_queue.get())

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
    max_workers: int = typer.Option(8, help='Maximum number of worker threads'),
):
    logger.info(f"Generating data for category: {AMB_UNANS_CATEGORY}. Number of squestions to sample {n2sample} per database. Output dir: {output_dir}")
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
        max_workers=max_workers,
    )

    # save the output to files
    output_jsonl_fp = os.path.join(output_dir, split, AMB_UNANS_CATEGORY) + ".jsonl"
    write_jsonl(nlines, output_jsonl_fp)
    logger.info(f"Great success. Output dir: {output_jsonl_fp}")


if __name__ == '__main__':
    typer.run(main)
