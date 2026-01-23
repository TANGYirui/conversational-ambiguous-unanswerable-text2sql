import pandas as pd 
import json 
import os
import random
import copy
from typing import List, Dict, Tuple
import typer
from loguru import logger
from tqdm import tqdm
from rapidfuzz import fuzz

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
    get_lexically_similar_cell_values_from_schema,
    load_spider_dev_train_data,
    sample_questions_by_database,
    parse_for_where,
    is_numeric,
)
from helpers import (
    add_fewshots_from_path,
    extract_string_list_from_xml_tags,
)
from litellm_helpers import (
    router_completion_with_ratelimit_retry,
)


AMB_UNANS_CATEGORY = "Nonexistent_Value"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "__cache__")  # use current dir as a cache dire
os.makedirs(CACHE_DIR, exist_ok=True)
IGNORE_CACHE = False
CURRENT_DIR = os.path.dirname(__file__)

set_random_seed(7)


def get_sql_cell_filter(line):
    where_tuples = parse_for_where(line['query'])
    text_tuples = []
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
def identify_additional_cell_values_to_remove(line, spider_root_dir, system_prompt="", few_shots=[]):
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
def generate_followup_sql_for_single_line(line, spider_root_dir, system_prompt, few_shots):
    # question = line['question']
    # given the primarily removed cell values, find other cell values from the same column
    # randomly sample a non NULL cell value and use it to generate a new SQL
    db_id = line['db_id']
    schema_modification = line['schemaModification']
    spider_database_dir = os.path.join(spider_root_dir, "database")
    
    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return None

    # removed tab, column, and cell
    remove_cell = schema_modification['removeCell'][0]
    target_tab, target_col, target_cell = remove_cell['table'], remove_cell['column'], remove_cell['value']

    # get all the cells from the same column
    cells_from_tab_col = online_db.get_cell_values(only_unique_value=True, ignore_table_column_casing=True)[target_tab.lower().strip()][target_col.lower().strip()]
    other_cells = [
        {"table": target_tab, "column": target_col, "value": cell.strip()}  # some table and column contains cell with space as prefix, e.g., in flights table and destAirport, we have cell values like {'table': 'FLIGHTS', 'column': 'DestAirport', 'value': ' RFK'}]
        for cell in cells_from_tab_col
        if (
            cell is not None 
            and cell.strip().lower() not in (target_cell.strip().lower(), None)
            and len(cell.strip()) >= 1  # preventing empty string being used as replacement cell value
            and (
                cell.strip().lower() not in line['query'].lower().split()
                and f"'{cell.strip().lower()}'" not in line['query'].lower().split()
            )
            # prevent include cell that is mentioned in a different part of the query, for example 
            # a user question may be: "What are the students' first names who have both cats and dogs as pets?"
            # if we remove dog from database and query, we shall ensure `cat` is not the new cell that we will use
            # we shall not result in such base SQLs:
            # "SELECT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat' INTERSECT SELECT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat'"
        )
    ]
    # identify the most disimilar cells from the column
    similarities = []
    for tcc in other_cells:
        similarity = fuzz.ratio(tcc['value'], target_cell)
        similarities.append(similarity)

    sorted_ind = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=False)
    sorted_tab_col_cells = [other_cells[i] for i in sorted_ind]

    # generate templated unanswerable explanation
    sql_query = line['query']
    new_sql_list = []
    for tcc in sorted_tab_col_cells:
        # TODO: improve the method for generating new SQL
        target_with_quote = f"'{target_cell}'"
        cell_with_quote = f"'{tcc['value']}'"
        if target_with_quote in sql_query:
            new_sql = sql_query.replace(target_with_quote, cell_with_quote)
        else:
            new_sql = sql_query.replace(target_cell, tcc['value'])
        new_sql_list.append({"SQL": new_sql})

    if not new_sql_list:
        return None

    if "ambiguousUnanswerableConversation" not in line:
        line['ambiguousUnanswerableConversation'] = {}

    line['ambiguousUnanswerableConversation'].update(
        {
            "output_response_with_followup_and_sql_raw": [],
            "output_response_with_followup_and_sql_parsed": new_sql_list[:5],
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
    unans_explanation = generate_template_nonexistent_value_message(schema_modification)
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

        print(user_msg)
        print("---" * 30)
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


def generate_template_nonexistent_value_message(schema_modification):
    removed_cell_info = schema_modification['removeCell']
    table = removed_cell_info[0]['table']
    column = removed_cell_info[0]['column']
    cell = removed_cell_info[0]['value']
    # TODO: determine whether it is good to include the column and table in the templated response.
    template = f"To answer the question, we need a cell value like `{cell}` from column `{column}` in table `{table}`. However, such value does not exist . Can you ask a different question?"
    # template = f"To answer the question, we need a cell value like `{cell}`. However, such value does not exist . Can you ask a different question?"
    return template


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
    if "input_conversation_with_placeholder_to_fill" not in line['ambiguousUnanswerableConversation']:
        logger.error(f"No input_conversation_with_placeholder_to_fill in line: {line['ambiguousUnanswerableConversation']}")
        return None
    conversation_with_template = line['ambiguousUnanswerableConversation']['input_conversation_with_placeholder_to_fill']
    conversation_with_followup = copy.deepcopy(conversation_with_template)

    # filter out invalid SQLs that cannot be run
    valid_sql_response_list = []
    for sql_res in line['ambiguousUnanswerableConversation']['output_response_with_followup_and_sql_parsed']:
        try:
            result = online_db.run_sql(sql=sql_res['DB EXPERT'])
            valid_sql_response_list.append(sql_res)
        except Exception as ex:
            logger.error(f"Error running SQL: {ex}")
            continue
    if not valid_sql_response_list:
        return None

    line['ambiguousUnanswerableConversation']['valid_ranked_response_with_followup_sql_parsed'] = valid_sql_response_list

    for followup_sql in line['ambiguousUnanswerableConversation']['valid_ranked_response_with_followup_sql_parsed']:
        conversation_with_followup[-2] = {
            "USER": followup_sql['USER'],
        }
        conversation_with_followup[-1] = {
            "DB EXPERT": followup_sql['DB EXPERT'],
        }
        break

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<conversation>\n\n{json.dumps(conversation_with_followup, indent=2)}\n\n</conversation>"
    # print(user_msg)
    # print("---" * 30)
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
    print(response)
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


def process_lines_of_interest(
    lines,
    spider_root_dir: str,
    n2sample=False,
):
    # identify questions with WHERE column and TEXT based filter
    # remove the value from the WHERE column and other columns that contain the cell value
    # [optional] use claude to identify other columns that shall be removed --> reviewing all cell values is impossible, perhaps add synonyms to remove
    # generate valid SQL by replacing the removed cell value with other cell value
    # reverse generate follow up question

    # follow-up question generation
    followup_question_few_shots_filename = "fewshots_generating_followup_question_from_sql.ipynb"
    followup_question_few_shots_path = os.path.join(CURRENT_DIR, followup_question_few_shots_filename)
    followup_question_few_shots, followup_question_system_prompt = add_fewshots_from_path(path_str=followup_question_few_shots_path, extension=".ipynb")

    # rephrase the templated explanation
    rephrase_templated_explanation_few_shots_filename = "fewshots_rephrasing_templated_unanswerable_response.ipynb"
    rephrase_templated_explanation_few_shots_path = os.path.join(CURRENT_DIR, rephrase_templated_explanation_few_shots_filename)
    rephrase_templated_explanation_few_shots, rephrase_templated_explanation_system_prompt = add_fewshots_from_path(path_str=rephrase_templated_explanation_few_shots_path, extension=".ipynb")

    nlines = []
    if n2sample:
        lines = sample_questions_by_database(lines=lines, n_question_per_db=n2sample)

    for line in tqdm(lines):
        # get sql where text filter
        line_with_text_filter = get_sql_cell_filter(line=line)
        if not line_with_text_filter:
            logger.info(f"No text filter found in line: {line['query']}")
            continue

        # identify additional cell values to remove that are similar to that needs to be deleted
        line_with_additional_cell_values_to_remove = identify_additional_cell_values_to_remove(
            line=line_with_text_filter,
            spider_root_dir=spider_root_dir,
            system_prompt="",
            few_shots=[],
        )
        if not line_with_additional_cell_values_to_remove:
            continue

        # generate followup SQL by replacing the removed cell value with other cell value
        # we will modify the database accordingly by adding schema modification values
        line_with_followup_sql = generate_followup_sql_for_single_line(
            line=line_with_text_filter,
            spider_root_dir=spider_root_dir,
            system_prompt="",
            few_shots=[],
        )
        if not line_with_followup_sql:
            continue

        # generate followup question reversely based on the SQL
        line_with_followup_question = generate_followup_question_for_single_line(
            line=line_with_followup_sql,
            spider_root_dir=spider_root_dir,
            system_prompt=followup_question_system_prompt,
            few_shots=followup_question_few_shots,
        )
        if not line_with_followup_question:
            continue

        # rephrase the templated explanation
        line_with_rephrased_explanation = rephrase_the_templated_explanation_single_line(
            line=line_with_followup_question,
            spider_root_dir=spider_root_dir,
            system_prompt=rephrase_templated_explanation_system_prompt,
            few_shots=rephrase_templated_explanation_few_shots,
        )
        if not line_with_rephrased_explanation:
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

    nlines = process_lines_of_interest(
        lines=split_all,
        spider_root_dir=spider_data_root_dir,
            n2sample=n2sample,
    )

    # save the results in files
    output_jsonl_fp = os.path.join(output_dir, split, AMB_UNANS_CATEGORY) + ".jsonl"
    logger.info(f"Output dir: {output_jsonl_fp}")
    write_jsonl(nlines, output_jsonl_fp)    


if __name__ == '__main__':
    typer.run(main)
    # TODO: for question `'What are flight numbers of flights arriving at Airport "APG"?'`, the airport code cell value somehow all contain a space before the first letter
