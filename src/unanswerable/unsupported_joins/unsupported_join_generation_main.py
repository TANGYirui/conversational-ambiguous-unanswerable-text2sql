import pandas as pd 
import json 
import os
import re
import copy
from typing import List, Dict, Tuple
import typer
from loguru import logger
from tqdm import tqdm

from custom_sql_engine import (
    DbWithModification,
)
from utils import (
    set_random_seed,
    format_schema_to_markdown,
    create_simple_message,
    read_jsonl_file,
    write_jsonl,
    sample_questions_by_database,
    parse_for_where,
    get_select_table_column_info,
)
from helpers import (
    add_fewshots_from_path,
    extract_string_list_from_xml_tags,
)
from litellm_helpers import (
    router_completion_with_ratelimit_retry,
)

set_random_seed(8)
AMB_UNANS_CATEGORY = "Unsupported_Join"


def collect_db_id(json_data):
    db_name = {}
    for data in json_data:
        db_name[data['db_id']] = 1
    return db_name


def get_org_db_name(db_name):
    match_list = [m.start() for m in re.finditer('_', db_name)]
    last_index = match_list[-1]
    filter_db_name = db_name[:last_index]
    return filter_db_name


def get_questions_with_join(spider_data):
    questions_with_join = []
    total_line = 0
    total_line_with_join = 0
    for data in spider_data:
        total_line += 1
        if "join" in data['query'].lower():
            questions_with_join.append(data)
            total_line_with_join += 1
    logger.info(f"Total line: {total_line}. Line with join: {total_line_with_join}")
    return questions_with_join


def write_json(data_list, filename):
    with open(filename, "w") as fout:
        json.dump(data_list, fout, indent=2)


def identify_first_join_columns(sql_query):
    join_columns_using_table_alias = identify_join_columns_with_table_alias(sql_query=sql_query)
    join_columns = {}
    if join_columns_using_table_alias:
        for alias, column in join_columns_using_table_alias.items():
            original_table_name = get_original_table_name(sql_query=sql_query, alias=alias)
            if original_table_name:
                join_columns[original_table_name] = column
    return join_columns


def identify_join_columns_with_table_alias(sql_query):
    """
    Identifies the columns used for joining in a SQL query with a JOIN clause.
    
    Args:
        sql_query (str): The SQL query to analyze.
    
    Returns:
        dict: A dictionary where the keys are the table names and the values are the column names used for joining.
    """

    # Split the SQL query into individual lines
    lines = sql_query.strip().split('\n')
    
    # Find the line with the JOIN clause
    join_line = next((line for line in lines if 'join' in line.lower()), None)
    if join_line is None:
        return {}
    
    # Extract the table names and column names from the JOIN clause
    pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
    match = re.search(pattern, join_line)
    if match:
        table1, col1, table2, col2 = match.groups()
        return {table1: col1, table2: col2}
    else:
        return {}


def get_original_table_name(sql_query, alias='T1'):
    """
    Extract the original table name for a given alias in an SQL query.
    
    :param sql_query: The SQL query as a string.
    :param alias: The alias to find the original table name for. Default is 'T1'.
    :return: The original table name or None if not found.
    """
    # Regular expression pattern to capture "table_name AS alias" or "table_name alias"
    pattern = re.compile(
        rf'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?{alias}\b'
        r'|'
        rf'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?{alias}\b',
        re.IGNORECASE
    )

    matches = pattern.findall(sql_query)
    if not matches:
        return None

    # Flatten the matches and remove empty strings
    matches = [match for match_pair in matches for match in match_pair if match]
    return matches[0] if matches else None


def extract_questions_requiring_join(
    spider_data_root_dir: str = typer.Option(
        '../../spider/dataset', help='The path to the Spider dataset'
    ),
    output_dir: str = typer.Option(
        '.vscode/output_2024_05', help='The path to the output directory'
    ),
    split: str = typer.Option('dev', help='The split to use: dev or train'),
):
    if split == "dev":
        spider_path = os.path.join(spider_data_root_dir, 'dev.json')
    elif split == "train":
        spider_path = os.path.join(spider_data_root_dir, 'train_spider.json')

    with open(spider_path, 'r') as infile:
        spider_data = json.load(infile)

    # get data with join
    spider_data_with_join = get_questions_with_join(spider_data=spider_data)
    # random.shuffle(spider_data_with_join)
    spider_data_modified = []
    for line in spider_data_with_join:
        join_columns = identify_first_join_columns(line['query'])
        if join_columns:
            line['ambiguousUnanswerableCategory'] = AMB_UNANS_CATEGORY
            line['schemaModification'] = {
                "removeColumn": [
                    {
                        "table": key, "column": val
                    }
                    for key, val in join_columns.items()
                ]
            }
            
            spider_data_modified.append(line)

    output_split_dir = os.path.join(output_dir, AMB_UNANS_CATEGORY)
    output_split_sub_dir = os.path.join(output_split_dir, "schema_modification")
    os.makedirs(output_split_sub_dir, exist_ok=True)
    write_json(spider_data_modified, os.path.join(output_split_sub_dir, f'{split}.json'))
    logger.info(f"Get {len(spider_data_modified)} modified data out of {len(spider_data)} questions from {split} data set.")
    return spider_data_modified


def generate_template_unsupported_join_message(schema_modification):
    removed_column = schema_modification['removeColumn']
    table_1 = removed_column[0]['table']
    table_1_join_key = removed_column[0]['column']
    table_2 = removed_column[1]['table']
    table_2_join_key = removed_column[1]['column']
    example_join_keys = [table_1_join_key]
    if table_2_join_key not in example_join_keys:
        example_join_keys.append(table_2_join_key)
    template = f"To answer the question, we need to join the table {table_1} and {table_2}. However, they do not share any common columns (e.g., {' or '.join(example_join_keys)}) that can be used for joining and therefore the question cannot be answered. Can you ask a question that does not require join?"
    return template


def generate_followup_question_for_split(lines, split, spider_root_dir, system_prompt, few_shots, output_dir=None):
    updated_lines = []
    spider_database_dir = os.path.join(spider_root_dir, "database")
    output_sub_dir = os.path.join(output_dir, AMB_UNANS_CATEGORY, "followup_question_with_sql")
    os.makedirs(output_sub_dir, exist_ok=True)
    output_fp = os.path.join(output_sub_dir, f"{split}_followup_question_with_sql.json")
    if os.path.exists(output_fp):
        lines_with_result = read_jsonl_file(output_fp)
        key2lines = {
            line['question'] + json.dumps(line['schemaModification']): line
            for line in lines_with_result
            if line.get("ambiguousUnanswerableConversation")
        }
        logger.info(f"Skip {split} followup question with sql generation.")
    else:
        key2lines = {}

    exception_count = 0
    with open(output_fp, "w") as fout:
        for line in tqdm(lines):
            question = line['question']
            schema_modification = line['schemaModification']
            key = question + json.dumps(schema_modification)

            if key in key2lines:
                nline = key2lines[key]
                updated_lines.append(nline)
                fout.write(json.dumps(nline) + "\n")
                continue

            db_id = line['db_id']
            try:
                online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
            except Exception as ex:
                exception_count += 1
                logger.error(f"Error loading database: {ex}")
                online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
                continue
            unans_explanation = generate_template_unsupported_join_message(schema_modification)
            schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))

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
                    "DB EXPERT": "FILL-IN-YOUR-SQL-RESPONSE-HERE"
                }
            ]

            user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<conversation>\n\n{json.dumps(conversation_list, indent=2)}\n\n</conversation>"
            tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
            response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
            try:
                result = json.loads(extract_string_list_from_xml_tags(response, "result")[0])
            except Exception as ex:
                logger.error(f"Error parsing result: {ex}")
                result = {}
            line['ambiguousUnanswerableConversation'] = {
                "input_conversation_with_placeholder_to_fill": conversation_list,
                "output_response_with_followup_and_sql_raw": response,
                "output_response_with_followup_and_sql_parsed": result,
            }
            updated_lines.append(line)
            fout.write(json.dumps(line) + "\n")
    logger.info(f"Exception count: {exception_count}")
    return updated_lines


def select_most_natural_followup_for_split(
    lines: List[Dict],
    split: str,
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict],
    output_dir: str,
):
    updated_lines = []
    spider_database_dir = os.path.join(spider_root_dir, "database")
    output_sub_dir = os.path.join(output_dir, AMB_UNANS_CATEGORY, "ranked_followup_question_with_sql")
    os.makedirs(output_sub_dir, exist_ok=True)
    output_fp = os.path.join(output_sub_dir, f"{split}_ranked_followup_question_with_sql.json")
    if os.path.exists(output_fp):
        lines_with_result = read_jsonl_file(output_fp)
        key2lines = {
            line['question'] + json.dumps(line['schemaModification']): line
            for line in lines_with_result
            if line.get("ambiguousUnanswerableConversation", {}).get("ranked_response_with_followup_sql_raw")
        }
        logger.info(f"Skip {split} followup question with sql generation.")
    else:
        key2lines = {}

    exception_count = 0
    with open(output_fp, "w") as fout:
        for line in tqdm(lines):
            question = line['question']
            schema_modification = line['schemaModification']
            key = question + json.dumps(schema_modification)

            if key in key2lines:
                nline = key2lines[key]
                updated_lines.append(nline)
                fout.write(json.dumps(nline) + "\n")
                continue

            db_id = line['db_id']
            try:
                online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
            except Exception as ex:
                exception_count += 1
                logger.error(f"Error loading database: {ex}")
                # online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
                continue

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
            updated_lines.append(line)
            fout.write(json.dumps(line) + "\n")
    logger.info(f"Exception count: {exception_count}")
    return updated_lines


def remove_invalid_followup_based_on_sql_execution(lines, spider_root_dir: str):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    updated_lines = []
    exception_count = 0
    for line in tqdm(lines):
        schema_modification = line['schemaModification']

        db_id = line['db_id']
        try:
            online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
        except Exception as ex:
            exception_count += 1
            logger.error(f"Error loading database: {ex}")
            continue
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
        updated_lines.append(line)
    return updated_lines


def rephrase_the_templated_explanation(
    lines,
    split: str,
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict],
    output_dir: str,
):
    updated_lines = []
    spider_database_dir = os.path.join(spider_root_dir, "database")
    output_sub_dir = os.path.join(output_dir, AMB_UNANS_CATEGORY, "rephrased_explanation_selected_followup_sql")
    os.makedirs(output_sub_dir, exist_ok=True)
    output_fp = os.path.join(output_sub_dir, f"{split}_rephrased_explanation_selected_followup_sql.json")
    if os.path.exists(output_fp):
        lines_with_result = read_jsonl_file(output_fp)
        key2lines = {
            line['question'] + json.dumps(line['schemaModification']): line
            for line in lines_with_result
            if line.get("ambiguousUnanswerableConversation", {}).get("rephrased_explanation_selected_followup_sql")
        }
        logger.info(f"Skip {split} followup question with sql generation.")
    else:
        key2lines = {}

    exception_count = 0
    with open(output_fp, "w") as fout:
        for line in tqdm(lines):
            question = line['question']
            schema_modification = line['schemaModification']
            key = question + json.dumps(schema_modification)

            if key in key2lines:
                nline = key2lines[key]
                updated_lines.append(nline)
                fout.write(json.dumps(nline) + "\n")
                continue

            db_id = line['db_id']
            try:
                online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
            except Exception as ex:
                exception_count += 1
                logger.error(f"Error loading database: {ex}")
                # online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
                continue
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
            updated_lines.append(line)
            fout.write(json.dumps(line) + "\n")
    return updated_lines


def is_removed_join_column_also_select_where_clause(line):
    """
    we delete columns that are used to join, however, if they are also used in the SELECT or WHERE clause. Then the question will be Nonexistent SELECT/WHERE column as well.
    """
    parsed_select_clause = get_select_table_column_info(line['query'])[0]
    parsed_where_clause = parse_for_where(line['query'])
    removed_columns = line['schemaModification']
    for tab_col in removed_columns['removeColumn']:
        tab, col = tab_col['table'], tab_col['column']
        for sql_tab_col in parsed_select_clause + parsed_where_clause:
            
            if (
                tab.lower().strip() == sql_tab_col['table'].lower().strip()
                and col.lower().strip() == sql_tab_col['column'].lower().strip()
            ):
                logger.info(f"Idenfied join columns also used in SELECT/WHERE clause. Question: {line['question']}. SQL: {line['query']}")
                return True
    return False


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
    current_dir = os.path.dirname(__file__)

    # extract questions with join and include metadata: ambiguousUnanswerableCategory & schemaModification
    if split == "all":
        dev_with_join = extract_questions_requiring_join(spider_data_root_dir=spider_data_root_dir, output_dir=output_dir, split="dev")
        train_with_join = extract_questions_requiring_join(spider_data_root_dir=spider_data_root_dir, output_dir=output_dir, split="train")
        split_with_join = dev_with_join + train_with_join
    else:
        split_with_join = extract_questions_requiring_join(spider_data_root_dir=spider_data_root_dir, output_dir=output_dir, split=split)

    if n2sample:
        split_with_join = sample_questions_by_database(lines=split_with_join, n_question_per_db=n2sample)

    # generate the conversation file for the modified data
    # come up with a natural question that can be answered with the clarification
    few_shots_filename = "fewshots_examples_generating_followup_sql.ipynb"
    few_shots_path = os.path.join(current_dir, few_shots_filename)
    few_shots, simple_system_prompt = add_fewshots_from_path(path_str=few_shots_path, extension=".ipynb")
    split_with_multiple_followup = generate_followup_question_for_split(
        lines=split_with_join,
        split=split,
        spider_root_dir=spider_data_root_dir,
        system_prompt=simple_system_prompt, few_shots=few_shots, output_dir=output_dir
    )

    # select the most natural follow-up questions
    selection_fewshots_filename = "fewshots_examples_selecting_most_natural_follow_up_refined_system_prompt.ipynb"
    selection_fewshots_path = os.path.join(current_dir, selection_fewshots_filename)
    selection_shots, selection_system = add_fewshots_from_path(selection_fewshots_path)
    split_with_selected_followup = select_most_natural_followup_for_split(
        lines=split_with_multiple_followup,
        split=split,
        spider_root_dir=spider_data_root_dir,
        system_prompt=selection_system,
        few_shots=selection_shots,
        output_dir=output_dir,
    )

    # check whether the SQL is executable
    split_with_multiple_followup_with_valid_sql = remove_invalid_followup_based_on_sql_execution(
        lines=split_with_selected_followup,
        spider_root_dir=spider_data_root_dir,
    )

    # rephrase the templated unanswerable explanation
    rephrase_fewshots_filename = "fewshots_examples_for_rephrasing_templated_unanswerable_response.ipynb"
    rephrase_fewshots_path = os.path.join(current_dir, rephrase_fewshots_filename)
    selection_shots, selection_system = add_fewshots_from_path(rephrase_fewshots_path)
    split_with_rephrase_selected_followup_sql = rephrase_the_templated_explanation(
        lines=split_with_multiple_followup_with_valid_sql,
        split=split,
        spider_root_dir=spider_data_root_dir,
        system_prompt=selection_system,
        few_shots=selection_shots,
        output_dir=output_dir,
    )

    # remove invalid lines where removed join columns are also used in SELECT/WHERE clause
    split_with_rephrase_selected_followup_sql = [
        line for line in split_with_rephrase_selected_followup_sql
        if not is_removed_join_column_also_select_where_clause(line)
    ]

    # save the results in files
    nlines = split_with_rephrase_selected_followup_sql
    output_jsonl_fp = os.path.join(output_dir, split, AMB_UNANS_CATEGORY) + ".jsonl"
    logger.info(f"Output dir: {output_jsonl_fp}")
    write_jsonl(nlines, output_jsonl_fp)    


if __name__ == '__main__':
    typer.run(main)
