import pandas as pd 
import json 
import os
import re
import json
from sqlglot import parse_one, exp, errors
import copy
from typing import List, Dict, Tuple
import typer
from loguru import logger
from tqdm import tqdm

from custom_sql_engine import (
    DbWithModification,
)
from utils import (
    format_schema_to_markdown,
    create_simple_message,
    read_jsonl_file,
    get_select_table_column_info,
    NO_ALIAS_PREFIX,
    parse_for_where,
    write_jsonl,
    sample_questions_by_database,
)
from helpers import (
    add_fewshots_from_path,
    extract_string_list_from_xml_tags,
)
from litellm_helpers import (
    router_completion_with_ratelimit_retry,
)


AMB_UNANS_CATEGORY = "Nonexistent_WHERE_Column"


def sql_identifier(s):
    return '"' + s.replace('"', '""') + '"'


def get_col_names(conn, table_name):
    all_col_info = conn.execute("PRAGMA table_info({})".format(sql_identifier(table_name)))
    primary_keys, col_names = [], []
    for col_info in all_col_info: 
        col_name = col_info[1]
        col_names.append(col_name)
        if col_info[-1] == 1:
            primary_keys.append(col_name)
    return col_names, primary_keys


def get_table_name_and_alias_mapping(parsed_sql):
    # Get all table names and aliases. E.g.,
    # for `SELECT * FROM table_a AS a, table_b AS b;`, `a` and `b` are table aliases for table_a and table_b.
    # for `'SELECT count(*) FROM singer'`, there are no aliases and we use `
    # another example with table alias is: `SELECT T1.last_name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id WHERE T2.age  =  ( SELECT max(age) FROM Dogs )`
    # we want to map: T1 to Owners, T2 to Dogs
    table_store_dict = dict()
    not_found_count = 0
    for table in parsed_sql.find_all(exp.Table):
        if str(table.alias) == '':
            table_alias = f'{NO_ALIAS_PREFIX}_{not_found_count + 1}'
            not_found_count += 1
        else:
            table_alias = table.alias.lower()
        table_store_dict[table_alias] = table.name
        # trivial case - table name maps to itself
        table_store_dict[table.name.lower()] = table.name
    return table_store_dict


def clean_string(name):
    # if len(name.split(' ')) > 1:
    #     logger.error(f"Name {name} has more than one word")
    # return name.split(' ')[0]
    return name.strip()


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def clean_where_filter_value(value_name, is_like_filter=False):
    # clean quote from regular cell value. E.g., `city = "France"`
    clean_value_name = value_name.strip('"').strip("'")
    if is_like_filter:
        # clean like filter: `"song_name LIKE '%Hey%'"`
        clean_value_name = clean_value_name.strip('%')
    return clean_value_name


def group_data_by_db_where_clause(lines, lower_case=True):
    db_wise_where_info = dict()
    for ctr, data in tqdm(enumerate(lines), total=len(lines)):
        all_condition_tuples = parse_for_where(data['query'])
        for condition_tuple in all_condition_tuples:
            # TODO: Get the values of the table
            db_id = data['db_id']
            if db_id not in db_wise_where_info:
                db_wise_where_info[db_id] = dict()
            # Store information
            if not lower_case:
                cnd_tuple_str = str(condition_tuple)
            else:
                cnd_tuple_str = str((condition_tuple[0].lower(), condition_tuple[1].lower()))
            if cnd_tuple_str not in db_wise_where_info[db_id]:
                db_wise_where_info[db_id][cnd_tuple_str] = []
            db_wise_where_info[db_id][cnd_tuple_str].append(data)
    return db_wise_where_info


def group_data_by_db_select_clause(lines, lower_case=True):
    db_wise_select_info = dict()
    for ctr, data in tqdm(enumerate(lines), total=len(lines)):
        sql_query = data['query']
        select_col_sets = get_select_table_column_info(sql_query=sql_query)

        # NOTE: we only focuses on the 1st select sets of select columns as later ones are often used in a nested way
        # it is hard to know whether the 2nd select COLUMN is used in a nested way or not
        # e.g., select name from dogs where dog_id not in ( select dog_id from treatments group by dog_id having sum(cost_of_treatment)  >  1000 )
        # e.g., SELECT first_name FROM Professionals UNION SELECT first_name FROM Owners EXCEPT SELECT name FROM Dogs
        select_cols = select_col_sets[0]
        if select_cols:
            # logger.info(f"Select columns identified for sql: `{data['query']}`. Columns: {select_cols}")
            pass
        else:
            if "*" not in sql_query:
                logger.warning(f"No select columns identified for sql: `{sql_query}`")
            pass

        # parsed_select_clause = parse_for_select(sql_query=data['query'])
        # parsed_sql = parse_one(data['query'])
        # select_clause = list(parsed_sql.find_all(exp.Select))
        # select_clause = str(parsed_sql.select)
        # if lower_case:
        #     select_clause = select_clause.lower()
        db_id = data['db_id']
        for select_tab_col in select_cols:
            select_tab_col_str = str((select_tab_col['table'].lower(), select_tab_col['column'].lower()))
            if db_id not in db_wise_select_info:
                db_wise_select_info[db_id] = dict()
            if select_tab_col_str not in db_wise_select_info[db_id]:
                db_wise_select_info[db_id][select_tab_col_str] = []
            db_wise_select_info[db_id][select_tab_col_str].append(data)
    return db_wise_select_info


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


def get_questions_with_where_clause(spider_data):
    questions_of_interest = []
    total_line = 0
    total_line_of_interest = 0
    for line in spider_data:
        total_line += 1
        parsed_where_clause = parse_for_where(line['query'])
        if parsed_where_clause:
            for single_where_clause in parsed_where_clause:
                line['ambiguousUnanswerableCategory'] = AMB_UNANS_CATEGORY
                line['schemaModification'] = {
                    "removeColumn": [
                        {
                            "table": single_where_clause['table'], "column": single_where_clause['column']
                        }
                    ]
                }
                questions_of_interest.append(line)
                total_line_of_interest += 1
        else:
            if "where" in line['query'].lower():
                logger.debug(f"No where clause: {line['query']}")
            # logger.debug(f"No where clause: {data['query']}")
            pass
    logger.info(f"Total line: {total_line}. Line with where Clause: {total_line_of_interest}")
    return questions_of_interest


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


def load_spider_dev_train_data(
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

    return spider_data


def generate_template_nonexistent_where_column_message(schema_modification):
    removed_column_info = schema_modification['removeColumn']
    table = removed_column_info[0]['table']
    column = removed_column_info[0]['column']

    template = f"To answer the question, we need filter using a column like `{column}` from table `{table}`. However, such column does not exist . Can you ask a different question?"
    return template


def generate_followup_question_for_split(lines, split, spider_root_dir, system_prompt, few_shots, output_dir=None, questions_groupby_db_and_select=None, use_LLM_as_backup=False):
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
                continue
            removed_tab_col = schema_modification['removeColumn'][0]

            # generate templated unanswerable explanation
            unans_explanation = generate_template_nonexistent_where_column_message(schema_modification)
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

            # get current question's select info
            questions_with_similar_select = []
            select_tab_col_list = get_select_table_column_info(sql_query=line['query'])[0]
            for select_tab_col in select_tab_col_list:
                select_tab_col_str = str((select_tab_col['table'].lower(), select_tab_col['column'].lower()))
                similar_question_list = questions_groupby_db_and_select.get(db_id, {}).get(select_tab_col_str, [])
                logger.debug(f"Current question: {line['question']}")
                for tmp_question in similar_question_list:
                    # a question is a valid follow-up if it has the same select info and different query
                    # and if the removed column does not occur in the SQL that answers the follow-up
                    if tmp_question['query'] != line['query'] and removed_tab_col['column'] not in tmp_question['query']:
                        questions_with_similar_select.append(tmp_question)
                    else:
                        if tmp_question['query'] == line['query']:
                            pass
                        elif removed_tab_col['column'] in tmp_question['query']:
                            logger.debug(f"Question not valid as followup: removed column `{removed_tab_col['column']}` is used in the question: `{tmp_question['query']}`")
                        else:
                            logger.debug(f"Question not valid as followup: {tmp_question['question']}")

            # randomly select a question and uses it as follow up question and SQL
            if questions_with_similar_select:
                result_list = [
                    {"USER": item['question'], "DB EXPERT": item['query']}
                    for item in questions_with_similar_select
                ]

                line['ambiguousUnanswerableConversation'] = {
                    "input_conversation_with_placeholder_to_fill": conversation_list,
                    "output_response_with_followup_and_sql_raw": json.dumps(result_list),
                    "output_response_with_followup_and_sql_parsed": result_list,
                }

            elif use_LLM_as_backup:
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
            else:
                logger.warning(f"Questions without followup question and SQL generated: {line['query']}")
                continue

            updated_lines.append(line)
            fout.write(json.dumps(line) + "\n")
    logger.info(f"Exception count: {exception_count}")
    logger.info(f"Total lines: {len(lines)}, updated lines with followup and SQL: {len(updated_lines)}")
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
            try:
                sql = followup_sql['DB EXPERT']
                if online_db.is_sql_executable(sql):
                    execution_result = online_db.run_sql(sql)
                    followup_sql['SQL_EXECUTION_RESULTS'] = execution_result
                    valid_followup_sql.append(followup_sql)
                else:
                    logger.warning(f"Invalid followup sql: {sql}")
            except Exception as ex:
                logger.error(f"Failed SQL execution check, error: {ex}. Followup sql: {followup_sql}. Detailed line for debugging:\n{json.dumps(line, indent=2)}")
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

    if split == "all":
        dev_all = load_spider_dev_train_data(spider_data_root_dir=spider_data_root_dir, output_dir=output_dir, split="dev")
        train_all = load_spider_dev_train_data(spider_data_root_dir=spider_data_root_dir, output_dir=output_dir, split="train")
        split_all = dev_all + train_all
    else:
        split_all = load_spider_dev_train_data(spider_data_root_dir=spider_data_root_dir, output_dir=output_dir, split=split)

    if n2sample:
        if n2sample <= 2:
            # TODO: generate follow-up question using LLM instead of sampling from the question pool.
            logger.error("For non-existent where column, we sample follow-up questions from the question pool within the same database. As a result, setting n2sample to 1 will not produce any data. Set it n2sample to at least 3.")
            logger.warning("To guarantee that there are at least some data generated, na2ample is change to 3.")
            n2sample = 3
            # raise Exception("For non-existent where column, we sample follow-up questions from the question pool within the same database. As a result, setting n2sample to 1 will not produce any data. Set it n2sample to at least 3.")
        split_all = sample_questions_by_database(lines=split_all, n_question_per_db=n2sample)

    # # extract questions with join and include metadata: ambiguousUnanswerableCategory & schemaModification
    lines_of_interest = get_questions_with_where_clause(spider_data=split_all)

    # organize data by database and select clause
    questions_groupby_db_and_select = group_data_by_db_select_clause(lines=split_all, lower_case=True)

    # # organize data by database and where clause
    # questions_groupby_db_and_where = group_data_by_db_where_clause(lines=split_all, lower_case=True)

    # generate the conversation file for the modified data
    # come up with a natural question that can be answered with the clarification
    # few_shots_filename = "fewshots_examples_user_db_expert.ipynb"
    # few_shots_path = os.path.join(current_dir, few_shots_filename)
    # few_shots, simple_system_prompt = add_fewshots_from_path(path_str=few_shots_path, extension=".ipynb")
    few_shots, simple_system_prompt = [], None
    split_with_multiple_followup = generate_followup_question_for_split(
        lines=lines_of_interest,
        split=split,
        spider_root_dir=spider_data_root_dir,
        output_dir=output_dir,
        questions_groupby_db_and_select=questions_groupby_db_and_select,
        system_prompt=simple_system_prompt, few_shots=few_shots,
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

    # save the results in files
    nlines = split_with_rephrase_selected_followup_sql
    output_jsonl_fp = os.path.join(output_dir, split, AMB_UNANS_CATEGORY) + ".jsonl"
    logger.info(f"Output dir: {output_jsonl_fp}")
    write_jsonl(nlines, output_jsonl_fp)    


if __name__ == '__main__':
    typer.run(main)
