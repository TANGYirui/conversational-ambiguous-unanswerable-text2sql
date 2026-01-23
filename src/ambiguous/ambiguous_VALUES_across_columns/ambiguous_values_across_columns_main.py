import pandas as pd 
import json 
import os
import random
import copy
from typing import List, Dict, Tuple
import typer
from loguru import logger
from tqdm import tqdm

from simple_cache import (
    cache_results
)

from custom_sql_engine import (
    DbWithModification,
)
from utils import (
    set_random_seed,
    format_schema_to_markdown,
    create_simple_message,
    write_jsonl,
    read_jsonl_file,
    get_select_table_column_info,
    get_lexically_similar_cell_values_from_schema,
    get_lexically_similar_columns_from_schema,
    load_spider_dev_train_data,
    get_all_table_column_info,
    sample_questions_by_database,
    parse_for_where,
)
from helpers import (
    add_fewshots_from_path,
    extract_string_list_from_xml_tags,
)
from litellm_helpers import (
    router_completion_with_ratelimit_retry,
)


AMB_UNANS_CATEGORY = "Ambiguous_VALUES_across_Columns"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "__cache__")  # use current dir as a cache dire
os.makedirs(CACHE_DIR, exist_ok=True)
IGNORE_CACHE = False
CURRENT_DIR = os.path.dirname(__file__)

set_random_seed(7)


@cache_results(CACHE_DIR, ignore_cache=IGNORE_CACHE)
def identify_where_column_for_removal(line, spider_root_dir):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    parsed_where_clause = parse_for_where(line['query'])
    if parsed_where_clause:
        # for single_where_clause in parsed_where_clause:
        #     line['ambiguousUnanswerableCategory'] = AMB_UNANS_CATEGORY
        #     line['schemaModification'] = {
        #         "removeColumn": [
        #             {
        #                 "table": single_where_clause[0], "column": single_where_clause[1]
        #             }
        #         ]
        #     }
        single_where_clause = random.choice(parsed_where_clause)
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=None)
        tab_col_cells = online_db.get_cell_values(only_unique_value=True)
        lexically_related_columns = get_lexically_similar_columns_from_schema(single_where_clause, tab_col_cells=tab_col_cells)
        line['ambiguousUnanswerableCategory'] = AMB_UNANS_CATEGORY
        line['schemaModification'] = {
            "removeColumn": [
                {
                    "table": single_where_clause['table'], "column": single_where_clause['column'], "value": single_where_clause['value'],
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
    line, spider_root_dir,
    system_prompt, few_shots,
):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    schema_modification = line.get("schemaModification", {})

    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return {}
    # generate templated unanswerable explanation
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
def generate_replacement_columns_for_single_line(
    line, spider_root_dir, 
    system_prompt, few_shots, 
):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    schema_modification = line.get("schemaModification", {})

    try:
        # we show the complete schema to help model find better replacement columns that will be ambiguous
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification={})
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return {}
    # generate templated unanswerable explanation
    schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<column>\n{json.dumps(schema_modification['removeColumn'][0])}\n</column>\n\n<question>\n{line['question']}\n</question>\n\n<sql>\n{line['query']}\n</sql>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
    # print(user_msg)
    # print(response)
    # print("---" * 20)
    try:
        result = json.loads(extract_string_list_from_xml_tags(response, "result")[0])
        if result[0]['table'] != result[1]['table']:
            logger.error(f"Model predicted replacement columns belong to different tables. Result:\n{json.dumps(result, indent=2)}")
            return None
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        result = {}

    if len(result) != 2:
        logger.info(f"Invalid results. Shall contain exactly two columns in the replacement. result: {result}")
        logger.info(f"Rmoeve column: {schema_modification}, Related columns: {result}")
        return None

    line['ambiguousUnanswerableConversation'] = {
            "replacementColumnCandidates": {
            "raw": response,
            "parsed": result
        }
    }

    # generate refined responses based on system prompt and refinement fewshots

    # extract the cell values from the column to delete
    tab_col_to_delete = schema_modification['removeColumn'][0]
    removed_table_lower = tab_col_to_delete['table'].lower().strip()
    removed_column_lower = tab_col_to_delete['column'].lower().strip()
    cell_values_with_type_all = online_db.get_cell_values(
        only_unique_value=False,
        ignore_table_column_casing=True,
        include_column_type=True,
    )
    if cell_values_with_type_all.get(removed_table_lower, {}).get(removed_column_lower, None) is None:
        logger.error(f"Column or table not found: {tab_col_to_delete}. The schema is: {online_db.get_schema()}")
        return None
    else:
        cell_values_with_type = cell_values_with_type_all[removed_table_lower][removed_column_lower]

    if not cell_values_with_type['cell_values']:
        logger.error(f"No cell values found for column: {tab_col_to_delete}. The schema is: {online_db.get_cell_values(only_unique_value=True)}")
        return None

    # mock cell values for replacement columns
    replacement_column_cells_to_add = []
    for tab_col in result:
        cell_values = cell_values_with_type['cell_values']
        
        tab_col_to_add = copy.deepcopy(tab_col)
        # TODO: what is the best way to determine what cell values to add to the column, how much overlap there shall be
        # add the common value to both columns
        tab_col_to_add['value'] = random.sample(cell_values, int(len(cell_values) / 2) + 1) + [tab_col_to_delete['value']]
        tab_col_to_add['type'] = cell_values_with_type['type']
        replacement_column_cells_to_add.append(tab_col_to_add)

    line['schemaModification']['addColumn'] = replacement_column_cells_to_add

    return line


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def generate_followup_sql_for_single_line(line, spider_root_dir, system_prompt, few_shots):
    # question = line['question']
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
    # tabl_col_sample_cells = online_db.get_cell_values(only_unique_value=True)
    # schema_preview = format_schema_to_markdown(tabl_col_sample_cells)
    # all_tab_col_in_sql = get_all_table_column_info(line['query'])
    # all_tab_col_str_list = [
    #     f"{tab_col['table']}_{tab_col['column']}".lower()
    #     for tab_col in all_tab_col_in_sql
    # ]

    # fix minor bug when single quote are accidentally added to the key
    if "'replacementColumnCandidates'" in line['ambiguousUnanswerableConversation']:
        line['ambiguousUnanswerableConversation']['replacementColumnCandidates'] = line['ambiguousUnanswerableConversation']["'replacementColumnCandidates'"]
    candidate_columns = line['ambiguousUnanswerableConversation']['replacementColumnCandidates']['parsed']

    # # replace the removed column in SQL with new column if they are of the same physical type
    # # generate the replacement SQL
    # original_tab_col_type_mapping = original_db.get_table_column_type_mapping(lower_case=True)
    # target_tab, target_col = schema_modification['removeColumn'][0]['table'], schema_modification['removeColumn'][0]['column']
    # target_col_type = original_tab_col_type_mapping.get(target_tab.lower(), {}).get(target_col.lower(), "UNKNOWN")
    # table_col_type_map = online_db.get_table_column_type_mapping(lower_case=True)
    # valid_columns = []
    # for cand_tab_col in candidate_columns:
    #     cand_tab, cand_col = cand_tab_col['table'], cand_tab_col['column']
    #     cand_tab_col_str_lower = f"{cand_tab}_{cand_col}".lower()
    #     cand_col_type = table_col_type_map.get(cand_tab.lower(), {}).get(cand_col.lower(), '')
    #     if cand_tab_col_str_lower not in all_tab_col_str_list and cand_col.lower() != target_col.lower():
    #         valid_columns.append(cand_tab_col)

    # generate new SQL by simply replacing the original column with one of the candidate columns
    new_sql_list = []
    for cand_col in candidate_columns:
        original_sql = copy.deepcopy(line['query'])
        original_column = schema_modification['removeColumn'][0]
        # grounded_original_tab_col = original_db.get_lower_case_table_column_to_original_table_column_mapping()[original_column['table'].lower()][original_column['column'].strip().lower()]
        grounded_original_tab_col = original_db.get_grounded_table_column(table=original_column['table'], column=original_column['column'])
        # grounded_tab_col = online_db.get_lower_case_table_column_to_original_table_column_mapping().get(cand_col['table'].lower(), {}).get(cand_col['column'].strip().lower(), None)
        grounded_tab_col = online_db.get_grounded_table_column(table=cand_col['table'], column=cand_col['column'])
        if not grounded_tab_col:
            logger.error(f"Table_Column {cand_col['table']}_{cand_col['column']} not found in schema. Cannot be grounded. Skip")
            continue

        if grounded_tab_col:
            replacement_column_str = grounded_tab_col['column']
        else:
            replacement_column_str = cand_col['column']
        original_column_str_list = set([original_column['column'], original_column['column'].lower(), original_column['column'].upper()])
        if grounded_original_tab_col:
            original_column_str_list.add(grounded_original_tab_col['column'])

        found_matching_original_column = False
        for original_column_str in original_column_str_list:
            # replace the column in the SQL with the new column
            if original_column_str in original_sql:
                new_sql = original_sql.replace(original_column_str, replacement_column_str)
                new_sql_list.append(
                    {
                        "SQL": new_sql,
                        "table": grounded_tab_col['table'],
                        "column": grounded_tab_col['column'],
                    }
                )
                found_matching_original_column = True
                break
            else:
                pass
        if not found_matching_original_column:
            logger.error(f"None of the column form found in SQL: {original_sql}. Column candidates: {original_column_str_list}")

    # # for column from the same table, we can do simple replacement, for column from other table, we will need to leverage LLM for that
    # # generate the candidate SQLs based on the valid columns
    # replacement_column_str = json.dumps(valid_columns, indent=2)
    # original_sql = line['query']
    # original_column_str = json.dumps(schema_modification['removeColumn'][0])
    # user_msg = f"<schema>\n{schema_preview}\n</schema>\n\n<original_sql>\n{original_sql}\n</original_sql>\n\n<original_column>\n{original_column_str}\n</original_column>\n\n<replacement_columns>\n{replacement_column_str}\n</replacement_columns>"

    # # print(user_msg)
    # # print("---" * 30)
    # tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    # response = bedrock_llm.call(messages=tmp_msgs, system=system_prompt)
    # try:
    #     result = json.loads(extract_string_list_from_xml_tags(response, "result")[0])
    # except Exception as ex:
    #     logger.error(f"Error parsing result: {ex}")
    #     result = {}
    response = json.dumps(new_sql_list, indent=2)
    line['ambiguousUnanswerableConversation'].update(
        {
            "output_response_with_followup_and_sql_raw": response,
            "output_response_with_followup_and_sql_parsed": new_sql_list,
        }
    )
    return line


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def generate_followup_question_for_single_line(
    line, spider_root_dir, system_prompt, few_shots):
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
    unans_explanation = generate_template_ambiguous_values_across_column_message(schema_modification)
    cell_values = online_db.get_oracle_cell_values_for_amb_values_across_column(only_unique_value=True)
    # cell_values = online_db.get_cell_values(only_unique_value=True)
    schema_preview = format_schema_to_markdown(cell_values)
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
        print(user_msg)
        print("---" * 20)
        print(response)
        print("===" * 20)
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


def generate_template_ambiguous_values_across_column_message(schema_modification):
    # removed_column_info = schema_modification['removeColumn']
    # table = removed_column_info[0]['table']
    # column = removed_column_info[0]['column']
    # template = f"To answer the question, we need a column like `{column}` from table `{table}`. However, such column does not exist . Can you ask a different question?"
    col1, col2 = schema_modification['addColumn'][0]['column'], schema_modification['addColumn'][1]['column']
    table = schema_modification['addColumn'][0]['table']
    template = f"The question is asking to filter the results using either column `{col1}` or `{col2}` in the {table} table. Please specify which of these two columns should be used as the filter criterion."
    return template


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def select_most_natural_followup_for_split(
    line,
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict],
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
        schema_preview = format_schema_to_markdown(online_db.get_oracle_cell_values_for_amb_values_across_column(only_unique_value=True))
        user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<conversation>\n\n{json.dumps(templated_conversation, indent=2)}\n\n</conversation>\n\n<follow-up>\n{json.dumps(candidate_follow_ups, indent=2)}\n</follow-up>"
        tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
        response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
        print(user_msg)
        print(response)
        print("---" * 30)
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
):
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
    print(user_msg)
    print("---" * 30)
    print(response)
    print("===" * 30)
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


def process_lines_of_interest(
    lines,
    spider_root_dir: str,
):
    # step 1: identify the SELECT columns from the SQL
    # step 2: randomly choose a SELECT column, preferrably not ID column for further processing 
        # TODO: add ambiguous column cases based on ORDER BY/GROUP BY columns
    # step 3: copy its cell values out, come up with two columns with identical meaning to the original SELECT column
    # add the two new columns to the database with identical cell values shuffled in order
    # final SQL is just choosing one of the newly created column
    # reverse generate the user question
    # rephrase the templated explanation

    # semantically equivalent columns identification
    alternative_columns_few_shots_filename = "fewshots_examples_identifying_alternative_columns.ipynb"
    alternative_columns_few_shots_path = os.path.join(CURRENT_DIR, alternative_columns_few_shots_filename)
    alternative_columns_few_shots, alternative_columns_system_prompt = add_fewshots_from_path(path_str=alternative_columns_few_shots_path, extension=".ipynb")

    # generating replacement columns
    replacement_col_fewshots_fn = "fewshots_examples_generating_replacement_columns_sql.ipynb"
    replacement_col_fewshots_path = os.path.join(CURRENT_DIR, replacement_col_fewshots_fn)
    replacement_col_fewshots, replacement_col_system_prompt = add_fewshots_from_path(path_str=replacement_col_fewshots_path, extension=".ipynb")

    # # follow-up question generation
    # followup_sql_few_shots_filename = "fewshots_examples_generating_followup_sql_conversation.ipynb"
    # followup_sql_few_shots_path = os.path.join(CURRENT_DIR, followup_sql_few_shots_filename)
    # followup_sql_few_shots, followup_sql_system_prompt = add_fewshots_from_path(path_str=followup_sql_few_shots_path, extension=".ipynb")

    # # follow-up SQL generation
    # followup_sql_few_shots_filename = "fewshots_generating_sql_from_replacement_columns.ipynb"
    # followup_sql_few_shots_path = os.path.join(CURRENT_DIR, followup_sql_few_shots_filename)
    # followup_sql_few_shots, followup_sql_system_prompt = add_fewshots_from_path(path_str=followup_sql_few_shots_path, extension=".ipynb")

    # follow-up question generation
    followup_question_few_shots_filename = "fewshots_generating_followup_question_from_sql.ipynb"
    followup_question_few_shots_path = os.path.join(CURRENT_DIR, followup_question_few_shots_filename)
    followup_question_few_shots, followup_question_system_prompt = add_fewshots_from_path(path_str=followup_question_few_shots_path, extension=".ipynb")

    # select most natural followup and SQL
    select_natural_few_shots_filename = "fewshots_examples_selecting_most_natural_follow_up_refined_system_prompt.ipynb"
    select_natural_few_shots_path = os.path.join(CURRENT_DIR, select_natural_few_shots_filename)
    select_natural_few_shots, select_natural_system_prompt = add_fewshots_from_path(path_str=select_natural_few_shots_path, extension=".ipynb")

    # rephrase the templated explanation
    rephrase_templated_explanation_few_shots_filename = "fewshots_examples_for_rephrasing_templated_ambiguous_response.ipynb"
    rephrase_templated_explanation_few_shots_path = os.path.join(CURRENT_DIR, rephrase_templated_explanation_few_shots_filename)
    rephrase_templated_explanation_few_shots, rephrase_templated_explanation_system_prompt = add_fewshots_from_path(path_str=rephrase_templated_explanation_few_shots_path, extension=".ipynb")

    nlines = []
    # lines = sample_questions_by_database(lines=lines, n_question_per_db=5)
    for line in tqdm(lines):
        # identify the select Clause for removal
        line_with_select_clause = identify_where_column_for_removal(line=line, spider_root_dir=spider_root_dir)
        if not line_with_select_clause:
            continue

        # identify the alternative semantically equivalent columns for deletion
        line_with_semantic_similar_columns = identify_alternative_columns_for_deletion_single_line(
            line=line_with_select_clause,
            spider_root_dir=spider_root_dir,
            system_prompt=alternative_columns_system_prompt,
            few_shots=alternative_columns_few_shots,
        )
        if not line_with_semantic_similar_columns:
            continue

        # TODO: check that additionally removed columns (based on semantic and lexical overlap) are not used in the SQL
        # add parsed semantically equivalent columns to columns for removal
        try:
            line_with_semantic_similar_columns = add_semantic_equivalent_columns_to_schema_modification_and_check_removed_col_in_sql(
                line=line_with_semantic_similar_columns,
                spider_root_dir=spider_root_dir,
            )
        except Exception as ex:
            logger.error(f"Failed to parse the semantic equivalent columns: {ex}")
            logger.error(f"Line that caused the error: {json.dumps(line_with_semantic_similar_columns, indent=2)}")
            line_with_semantic_similar_columns = None
        if not line_with_semantic_similar_columns:
            continue

        # generate replacement columns and modify the SQL accordingly
        line_with_replacement_columns = generate_replacement_columns_for_single_line(
            line=line_with_semantic_similar_columns,
            spider_root_dir=spider_root_dir,
            system_prompt=replacement_col_system_prompt,
            few_shots=replacement_col_fewshots, 
        )
        if not line_with_replacement_columns:
            continue

        # generate follow-up SQL based on columns
        line_with_follow_up_sql = generate_followup_sql_for_single_line(
            line=line_with_replacement_columns,
            spider_root_dir=spider_root_dir,
            # system_prompt=followup_sql_system_prompt,
            # few_shots=followup_sql_few_shots, 
            system_prompt=None,
            few_shots=[],
        )
        if not line_with_follow_up_sql:
            continue

        # generate follow-up question via reverse generation
        line_with_followup_question_sql = generate_followup_question_for_single_line(
            line=line_with_follow_up_sql,
            spider_root_dir=spider_root_dir,
            system_prompt=followup_question_system_prompt,
            few_shots=followup_question_few_shots,
        )
        if not line_with_followup_question_sql:
            continue

        # select most natural follow-up sql and question
        line_with_ranked_followup = select_most_natural_followup_for_split(
            line=line_with_followup_question_sql,
            spider_root_dir=spider_root_dir,
            system_prompt=select_natural_system_prompt,
            few_shots=select_natural_few_shots,
            )
        if line_with_ranked_followup is None:
            continue

        # TODO: remove the invalid sql from candidates
        try:
            line_with_ranked_followup = remove_invalid_followup_based_on_sql_execution(line=line_with_ranked_followup, spider_root_dir=spider_root_dir)
        except Exception as ex:
            logger.error(f"Exceptions in removing invalid SQLs: {ex}")
            logger.error(f"Current line content for debugging:\n{json.dumps(line_with_ranked_followup, indent=2)}")
            continue

        # rephrase the templated explanation
        line_with_rephrased_explanation = rephrase_the_templated_explanation_single_line(
            line=line_with_ranked_followup,
            spider_root_dir=spider_root_dir,
            system_prompt=rephrase_templated_explanation_system_prompt,
            few_shots=rephrase_templated_explanation_few_shots,
            )
        if line_with_rephrased_explanation is None:
            continue

        nlines.append(line_with_rephrased_explanation)

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
    )

    output_jsonl_fp = os.path.join(output_dir, split, AMB_UNANS_CATEGORY) + ".jsonl"
    logger.info(f"Output dir: {output_jsonl_fp}")
    write_jsonl(nlines, output_jsonl_fp)


if __name__ == '__main__':
    # TODO: fixed the fewshots example with new templated ambiguosu explanation
    # TODO: add a filtering step to remove un-natural conversations
    typer.run(main)
