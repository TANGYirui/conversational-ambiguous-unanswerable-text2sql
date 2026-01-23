import json
import re
import os
from typing import List, Dict, Tuple
from sqlglot import parse_one, exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.scope import build_scope, find_all_in_scope
from rapidfuzz import fuzz

import typer
import copy
from loguru import logger
import random
import numpy as np


NO_ALIAS_PREFIX = "NO_ALIAS"


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
    # if name.startswith("'"):
    #     return clean_string(name[1:])
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


def get_all_table_column_info(sql_query: str) -> List[Dict]:
    all_where_columns = get_where_table_column_info(sql_query)
    all_select_columns = get_select_table_column_info(sql_query)
    all_columns = all_where_columns + [subitem for item in all_select_columns for subitem in item]
    unique_columns = []
    unique_column_strs = set()
    for tab_col in all_columns:
        tab_col_str = f"{tab_col['table']}_{tab_col['column']}"
        if tab_col_str not in unique_column_strs:
            unique_column_strs.add(tab_col_str)
            unique_columns.append(tab_col)
    return unique_columns


def get_all_condition_pairs(alias2table, column2table, where):
    # condition_str = where.strip('WHERE ') 
    # strip will result in edge cases. E.g., 'WHERE Ref_Shipping_Agents.shipping_agent_name = "USPS"'
    # after stripping `where`, the previous SQL section will become `'ef_Shipping_Agents.shipping_agent_name = "USPS"'`
    # NOTE: the parsing will fail for complex where condition like `T1.price - T1.cost > 0`
    if where.startswith(("WHERE ", "where ", "Where ")):
        condition_str = where[6:]
    else:
        condition_str = where
    
    condition_str_split_or = condition_str.split(" OR ")
    all_conditions = []
    for condition in condition_str_split_or:
        condition_str_split_or_and = condition.split(" AND ")
        all_conditions.extend(condition_str_split_or_and)
    condition_tuples = []
    condition_keywords = ["like", "LIKE", "<>", ">=", "<=", "=", ">", "<", ]
    for condition in all_conditions:
        for cond_kw in condition_keywords:
            if cond_kw in condition:
                condition_split = condition.split(cond_kw)
                col_name, value_name = condition_split[0], condition_split[1]
                clean_col_name, clean_value_name = clean_string(col_name), clean_string(value_name)
                is_like_filter = "like" == cond_kw.lower()
                clean_value_name = clean_where_filter_value(value_name=clean_value_name, is_like_filter=is_like_filter)
                clean_value_name = clean_value_name.strip('"').strip("'")  # remove the quote around cell values
                if '.' in clean_col_name:
                    # col_name has dot `.` in them when SQL use table alias. E.g., `T1.revenue` OR `WHERE T1.state = 'Virginia'`
                    # logger.debug(f"Column name contains dot: {clean_col_name}. where clause: {where}")
                    try:
                        table_alias, column_name = clean_col_name.split('.')
                        table_name = alias2table[table_alias.lower().strip()]
                    except KeyError:
                        logger.error(alias2table, clean_col_name)
                        continue
                    except Exception as ex:
                        logger.error(f"Failed to parse the condtion: {ex}. The condition is: {condition}. The where clause for debugging is: {where}.")
                        continue
                else:
                    try: 
                        column_name = clean_col_name
                        # table_alias = 'Super_1'  # TODO: ask Intern what this part of the code is for
                        # table_alias = f"{NO_ALIAS_PREFIX}_1"
                        table_name = column2table[column_name.strip().lower()]
                    except KeyError: 
                        # select the first table name in the dict 
                        # table_name = list(table_store_dict.values())[0]
                        logger.error(f"Failed to find the table for current column: {column_name}")
                        continue

                # condition_tuple = (table_name, column_name)
                # condition_tuples.append(condition_tuple)
                condition_dict = {"table": table_name, "column": column_name, "operator": cond_kw, "value": clean_value_name}
                condition_tuples.append(condition_dict)
                # if a condition tuple has been parsed, parse next conditions
                break
    # return list(set(condition_tuples))
    return condition_tuples


def parse_for_where(sql_query):
    parsed_sql = parse_one(sql_query)  
    # table_store_dict = get_table_name_and_alias_mapping(parsed_sql)
    alias2table = get_table_alias_to_name_map_from_sql(sql_query=sql_query)
    column2table = get_column_to_table_map_from_sql(sql_query=sql_query)
    all_condition_tuples = []
    for where in parsed_sql.find_all(exp.Where):
        string_where = str(where)
        # TODO: whether to only consider where clause with EQUALITY filter
        if (
            'select' not in string_where.lower()
            and (
                '=' in string_where  # equal filter
                or ">" in string_where  # larger or larger qual
                or "<" in string_where  # smaller or smaller qual
                or '<>' in string_where  # not equal filter
                or 'like' in string_where.lower()  # like filter
            )
        ):
            # ignoring cases where there are nested queries
            condition_tuples = get_all_condition_pairs(alias2table=alias2table, column2table=column2table, where=string_where)
            all_condition_tuples.extend(condition_tuples)

    return all_condition_tuples


def extract_nested_sql(sql_query: str) -> list:
    """
    Extracts all nested SQL queries from a given SQL statement.
    
    Args:
        sql_query (str): The SQL statement to extract nested queries from.
        
    Returns:
        list: A list of all nested SQL queries found in the input SQL statement.
    """
    nested_sql_queries = []
    stack = []

    for i, char in enumerate(sql_query):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                end = i + 1
                nested_sql = sql_query[start:end]
                if is_valid_sql(nested_sql.strip()):
                    nested_sql_queries.append(nested_sql.strip())
                else:
                    # Recursively search for nested SQL queries within the current match
                    # WARNING: Currently only handle single layer of nesting
                    # nested_sql_queries.extend(extract_nested_sql(nested_sql.strip()))
                    pass
    remove_parathesis_sql_quries = []
    for sql in nested_sql_queries:
        start_ind = 0
        end_ind = len(sql)
        if sql.startswith("("):
            start_ind = 1
        if sql.endswith(")"):
            end_ind = end_ind - 1
        # parathesis is also used for aggregation, we shall exclude these
        sql_only = sql[start_ind:end_ind]
        if sql_only.strip() == "*" or len(sql_only.split()) <= 1:
            logger.info(f"Exclude invalid sub sqls: {sql_only}. Before cleaning: {sql}. Complete SQL: {sql_query}")
            continue
        remove_parathesis_sql_quries.append(sql_only)
    return remove_parathesis_sql_quries


def is_valid_sql(sql_query: str) -> bool:
    """
    Checks if a given SQL statement is a valid query.
    
    Args:
        sql_query (str): The SQL statement to check.
        
    Returns:
        bool: True if the SQL statement is valid, False otherwise.
    """
    try:
        # You can use a SQL parsing library like sqlglot to validate the SQL query
        parse_one(sql_query)
        return True
    except:
        return False


def get_column_to_table_map_from_sql(sql_query: str) -> Dict[str, str]:
    parsed_sql_ast = parse_one(sql_query)

    try:
        qualify(parsed_sql_ast)
    except OptimizeError as ex:
        logger.warning(f"Failed to optimize the SQL. Column to table mapping may be inaccurate. Exception: {ex}")
    root = build_scope(parsed_sql_ast)

    # TODO: find nested sub query within the SQL and parse it
    # e.g., "SELECT col1, (SELECT Col2 FROM TABLE10 WHERE id = 1) AS nested_col FROM Table1"
    # we want to parse mapping between Col2 and TABLE10 as well

    # `find_all_in_scope` is similar to `Expression.find_all`, except it doesn't traverse into subqueries
    alias2table = {}
    column2table = {}
    for column in find_all_in_scope(root.expression, exp.Column):
        # col_str = str(column)
        # table_str, col_str = col_str.split('.')
        table_str, col_str = column.table, column.name
        if not column.table:
            continue
        try:
            table_alias = root.sources[column.table]
        except Exception as ex:
            logger.error(f"Get column to table map from SQL failed with error `{ex}` for column {column} in SQL: {sql_query}")
            continue
        alias_str = table_alias.alias
        alias2table[alias_str] = table_alias.name
        table_str = alias2table[alias_str]
        if col_str in column2table:
            # some column may appear in several tables with different meanings
            # e.g., T1.stadium_id == T2.stadium_id
            logger.warning(f"Column {col_str} already exists in column2table")
            if column2table[col_str] != table_str:
                if not isinstance(column2table[col_str], list):
                    column2table[col_str] = [column2table[col_str]]
                    logger.warning(f"Column {col_str} already exists in column2table")
                column2table[col_str].append(table_str)
            continue
        else:
            column2table[col_str] = table_str

    sub_sql_list = extract_nested_sql(sql_query)
    for sub_sql in sub_sql_list:
        column2table.update(get_column_to_table_map_from_sql(sql_query=sub_sql))

    # add an error catch for cases where INTERSECT/EXCEPT and nesting happens at the same time
    # if a SQL has nesting, it will be broken down at the above step, here, we shall only handle the broken down ones
    # e.g.,
    # 'SELECT COUNT(*) FROM (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "Dutch")'
    if not sub_sql_list:
        special_kws = ["intersect", "except"]
        for kw in special_kws:
            kw_with_space = f" {kw} "
            if kw_with_space in sql_query.lower():
                sub_quries = sql_query.lower().split(kw_with_space)
                for sq in sub_quries:
                    sub_mapping = get_column_to_table_map_from_sql(sql_query=sq)
                    column2table.update(sub_mapping)

    return column2table


def _get_table_alias_to_name_map_from_sql(sql_query: str) -> Dict[str, str]:
    # for `SELECT * FROM table_a AS a, table_b AS b;`, `a` and `b` are table aliases for table_a and table_b.
    # for `'SELECT count(*) FROM singer'`, there are no aliases and we use `
    # another example with table alias is: `SELECT T1.last_name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id WHERE T2.age  =  ( SELECT max(age) FROM Dogs )`
    # we want to map: T1 to Owners, T2 to Dogs
    
    parsed_sql_ast = parse_one(sql_query)
    try:
        qualify(parsed_sql_ast)
    except OptimizeError as ex:
        logger.warning(f"Failed to optimize the SQL. Table alias mapping may be inaccurate. Exception: {ex}")
    # fails to add table name before the column for SQL: `'SELECT name ,  country ,  age FROM singer ORDER BY age DESC'`
    root = build_scope(parsed_sql_ast)

    # TODO: find nested sub query within the SQL and parse it
    # e.g., "SELECT col1, (SELECT Col2 FROM TABLE10 WHERE id = 1) AS nested_col FROM Table1"
    # we want to parse mapping between Col2 and TABLE10 as well

    # `find_all_in_scope` is similar to `Expression.find_all`, except it doesn't traverse into subqueries
    alias2table = {}
    # only do the parsing if root is not None
    # bug fix for error: 
        #    for column in find_all_in_scope(root.expression, exp.Column):
        # AttributeError: 'NoneType' object has no attribute 'expression'
    if root:  
        for column in find_all_in_scope(root.expression, exp.Column):
            if column.table:
                try:
                    table_alias = root.sources[column.table]
                    alias_str = table_alias.alias
                    alias2table[alias_str] = table_alias.name
                except Exception as ex:
                    logger.error(f"Error `{ex}` in getting table alias to name mapping for column: {column}. Root: {root}. SQL query: {sql_query}")

    sub_sql_list = extract_nested_sql(sql_query)
    for sub_sql in sub_sql_list:
        alias2table.update(_get_table_alias_to_name_map_from_sql(sql_query=sub_sql))

    # add an error catch for cases where INTERSECT/EXCEPT and nesting happens at the same time
    # if a SQL has nesting, it will be broken down at the above step, here, we shall only handle the broken down ones
    # e.g.,
    # 'SELECT COUNT(*) FROM (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "Dutch")'
    if not sub_sql_list:
        special_kws = ["intersect", "except"]
        for kw in special_kws:
            kw_with_space = f" {kw} "
            if kw_with_space in sql_query.lower():
                sub_quries = sql_query.lower().split(kw_with_space)
                for sq in sub_quries:
                    sub_mapping = _get_table_alias_to_name_map_from_sql(sql_query=sq)
                    alias2table.update(sub_mapping)

    # add alias mapping from old method and combine the results
    alias2table_mapping = get_table_name_and_alias_mapping(parsed_sql=parse_one(sql=sql_query))
    alias2table_mapping.update(alias2table)

    return alias2table_mapping


def get_table_alias_to_name_map_from_sql(sql_query):
    try:
        alias2table_mapping = _get_table_alias_to_name_map_from_sql(sql_query=sql_query)
    except OptimizeError as ex:
        logger.warning(f"OptimizeError: {sql_query}. Exception trace: {ex}")
        alias2table_mapping = {}
    alias2table = get_table_name_and_alias_mapping(parsed_sql=parse_one(sql_query))
    alias2table.update(alias2table_mapping)
    return alias2table


def get_where_table_column_info(sql_query):
    parsed_sql = parse_one(sql_query)  
    # table_store_dict = get_table_name_and_alias_mapping(parsed_sql)
    alias2table = get_table_alias_to_name_map_from_sql(sql_query=sql_query)
    column2table = get_column_to_table_map_from_sql(sql_query=sql_query)
    all_condition_tuples = []
    for where in parsed_sql.find_all(exp.Where):
        string_where = str(where)
        # TODO: whether to only consider where clause with EQUALITY filter
        if (
            'select' not in string_where.lower()
            and (
                '=' in string_where  # equal filter
                or ">" in string_where  # larger or larger qual
                or "<" in string_where  # smaller or smaller qual
                or '<>' in string_where  # not equal filter
                or 'like' in string_where.lower()  # like filter
            )
        ):
            # ignoring cases where there are nested queries
            # condition_tuples = get_all_condition_pairs(table_store_dict, string_where)
            condition_tuples = get_all_condition_pairs(alias2table=alias2table, column2table=column2table, where=string_where)
            all_condition_tuples.extend(condition_tuples)

    # return list(set(all_condition_tuples))
    return all_condition_tuples


def get_db_expert_response(conversation):
    try:
        list_dict = json.loads(conversation)
        clarify = list_dict[2]['DB EXPERT']
    except:  # noqa:
        occurence = [m.start() for m in re.finditer("DB EXPERT", conversation)]
        search_string = conversation[occurence[1]:]
        first_end = search_string.find('}')
        clarify = search_string[len("DB EXPERT: "): first_end]
    return clarify


def generate_conversation_helpful(row, clarify):
    conversation = [
        {"DB EXPERT": "Hello, I am the Database Expert at your service. Please ask me any questions you have about the schema."},
        {"USER": row['Question']},
        {"DB EXPERT": clarify},
        {"DB EXPERT": row['Helpful SQL']}
    ]
    return conversation


def un_escape_string(escaped_string):
    try:
        escaped_string = json.loads(escaped_string)
    except:  # noqa
        escaped_string = escaped_string
    try:
        escaped_string = eval(escaped_string)
    except:  # noqa
        escaped_string = escaped_string
    return escaped_string


def clean_single_turn_conv(conv_str):
    if isinstance(conv_str, str):
        conv_str = un_escape_string(conv_str)
        if isinstance(conv_str, str):
            return conv_str.strip()
        else:
            return conv_str
    elif isinstance(conv_str, list):
        return conv_str


def clean_list_conv_in_dict(conv_list):
    clean_conv_list = []
    for key_val in conv_list:
        key, val = list(key_val.items())[0]
        clean_conv_list_row = {}
        clean_conv_list_row[key] = clean_single_turn_conv(val)
        clean_conv_list.append(clean_conv_list_row)
    return clean_conv_list


def create_simple_message(message, role, message_type="claude"):
    if message_type == "claude":
        msg = {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": message
                }
            ]
        }
    elif message_type == "litellm":
        msg = {
            "role": role,
            "content": message,
        }
    return msg


def format_schema_to_markdown(schema: Dict, num_sample_cell_to_show=3):
    markdown = ""
    for table_name, table_data in schema.items():
        markdown += f"## {table_name}\n\n"
        markdown += "| Column Name | Data Type | Description |\n"
        markdown += "| --- | --- | --- |\n"
        for column_name, column_data in table_data.items():
            data_type = type(column_data[0]).__name__ if column_data else "Unknown"
            description = f"Example values: {', '.join(map(str, column_data[:num_sample_cell_to_show]))}" if column_data else "No description available"
            markdown += f"| {column_name} | {data_type} | {description} |\n"
        markdown += "\n"
    return markdown


def read_json_file(file_path):
    """
    Reads a JSON file and returns its contents as a Python object.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: The contents of the JSON file as a Python object (dictionary or list).
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{file_path}'.")


def read_jsonl_file(file_path: str):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(json.loads(line))
    return lines


def write_jsonl(data_list, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for data in data_list:
            f.write(json.dumps(data) + '\n')


def write_json(data_list, filename):
    with open(filename, "w") as fout:
        json.dump(data_list, fout, indent=2)


def clean_select_col_name(name):
    '''
    Clean the table and column name from all bracketts 
    '''
    return name.split('(')[-1].split(')')[0].split(' ')[-1]
        

def get_select_table_column_info(sql_query):
    '''
    Input - SQL query 
    Output - Column names and correspoding table names in the SELECT Clause
    '''
    parsed_sql = parse_one(sql_query)

    # Get all table names and aliases
    table_store_dict = dict()
    not_found_count = 0
    for table in parsed_sql.find_all(exp.Table):
        if str(table.alias) == '':
            # table_alias = 'Super_{:d}'.format(not_found_count + 1)
            table_alias = f"{NO_ALIAS_PREFIX}_{not_found_count + 1}"
            not_found_count += 1
        else:
            table_alias = table.alias.lower()
        table_store_dict[table_alias] = table.name
        # trivial case - table name maps to itself
        table_store_dict[table.name.lower()] = table.name.lower()
    
    # print(table_store_dict)

    # Collect column names
    select_columns_all = []
    select_count = 0
    for select in parsed_sql.find_all(exp.Select):
        select_columns = []
        select_count += 1
        for projection in select.expressions:
            column_full_name = clean_select_col_name(str(projection))
            # print(column_full_name)
            if '*' in column_full_name:
                continue
            if '.' in column_full_name:
                try:
                    table_alias, column_name = column_full_name.split('.')
                    table_name = table_store_dict[table_alias.lower()]
                except KeyError:
                    logger.error(table_store_dict, column_full_name)
                    continue
            else:
                column_name = column_full_name
                try: 
                    # table_alias = 'Super_{:d}'.format(select_count)
                    table_alias = f"{NO_ALIAS_PREFIX}_{select_count}"
                    table_name = table_store_dict[table_alias]
                except KeyError: 
                    # select the first table name in the dict 
                    table_name = list(table_store_dict.values())[0]
            # res = (column_name.lower(), table_name.lower())
            # res = (column_name, table_name)
            res = {"column": column_name, "table": table_name}
            select_columns.append(res)
        select_columns_all.append(select_columns)
    # print("select_count is {:d}".format(select_count))
    return select_columns_all


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


def get_unique_select_clause(select_clause_list: List[Dict]):
    unique_select_clause = []
    unique_select_clause_str = []
    for select in select_clause_list:
        tmp_select_clause_str = f"{select['table']}_{select['column']}"
        if tmp_select_clause_str not in unique_select_clause_str:
            unique_select_clause_str.append(tmp_select_clause_str)
            unique_select_clause.append(select)
    return unique_select_clause_str


def get_lexically_similar_columns_from_schema(tab_col, tab_col_cells):
    special_col_names = ["name", "id", "first_name", "last_name", "identifier"]  # these two column names are table dependent and are only similar if  they are the same table
    similar_tab_cols = []
    # TODO: add Claude based similar column removal
    for tab, col_cells in tab_col_cells.items():
        for col in col_cells:
            # exact same column name
            if col.lower().strip() == tab_col['column'].lower().strip():
                if tab_col['table'].lower() == tab.lower():
                    # same table and column, does not repeatly add, continue
                    continue
                elif tab_col['table'].lower() != tab.lower() and col.lower().strip() in special_col_names:
                    # same column, different table, special case, does not add to similar_tab_cols, continue
                    # e.g., country and museum table may both contain `name` and `id` column but their meanings are probably different and shall not be considered as similar
                    continue
                else:
                    # logger.info(f"Column `{tab_col['column']}` from table `{tab_col['table']}` is identical to column `{col}` from table `{tab}`")
                    similar_tab_cols.append((tab, col))
                    continue

            # fuzz match
            if fuzz.ratio(col.lower().strip(), tab_col['column'].lower().strip()) > 90:
                similar_tab_cols.append((tab, col))
                logger.info(f"Column `{tab_col['column']}` from table `{tab_col['table']}` is similar to column `{col}` from table `{tab}`")
                continue
    # dedupe the columns
    similar_tab_cols = list(set(similar_tab_cols))
    similar_tab_cols = [
        {"table": item[0], "column": item[1]} for item in similar_tab_cols
    ]
    return similar_tab_cols


def get_lexically_similar_cell_values_from_schema(tab_col_cell, tab_col_cells):
    # special_col_names = ["name", "id", "first_name", "last_name", "identifier"]  # these two column names are table dependent and are only similar if  they are the same table
    similar_tab_col_cells = []
    # TODO: add Claude based similar column removal
    for tab, col_cells in tab_col_cells.items():
        for col, cells in col_cells.items():
            for cell in cells:
                # if cell is of numeric type, skip the comparison
                if is_numeric(cell) or cell is None:
                    continue
                # exact same column name
                if col.lower().strip() == tab_col_cell['column'].lower().strip():
                    if cell.strip().lower() == tab_col_cell['value'].strip().lower():
                        similar_tab_col_cells.append(
                            {
                                "table": tab, "column": col, "value": cell
                            }
                        )
                        logger.info(f"Cell value {cell} is identical to primary cell value {tab_col_cell['value']} from column {col} and table {tab}")
                        continue
                    elif fuzz.ratio(cell.strip().lower(), tab_col_cell['value'].strip().lower()) > 90:
                        similar_tab_col_cells.append(
                            {
                                "table": tab, "column": col, "value": cell
                            }
                        )
                        logger.info(f"Cell value {cell} is fuzz similar to primary cell value {tab_col_cell['value']} from column {col} and table {tab}")
                        continue
                else:
                    if cell.strip().lower() == tab_col_cell['value'].strip().lower():
                        similar_tab_col_cells.append(
                            {
                                "table": tab, "column": col, "value": cell
                            }
                        )
                        logger.info(f"Cell value {cell} is identical to primary cell value {tab_col_cell['value']} from column {col} and table {tab}")
                        continue
                    elif fuzz.ratio(cell.strip().lower(), tab_col_cell['value'].strip().lower()) > 90:
                        similar_tab_col_cells.append(
                            {
                                "table": tab, "column": col, "value": cell
                            }
                        )
                        logger.info(f"Cell value {cell} is fuzz similar to primary cell value {tab_col_cell['value']} from column {col} and table {tab}")
                        continue

    # dedupe the columns
    unique_tab_col_cells = []
    unique_tab_col_cell_strs = []
    for tab_col_cell in similar_tab_col_cells:
        tab_col_cell_str = f"{tab_col_cell['table']}_{tab_col_cell['column']}_{tab_col_cell['value']}"
        if tab_col_cell_str not in unique_tab_col_cell_strs:
            unique_tab_col_cells.append(tab_col_cell)
            unique_tab_col_cell_strs.append(tab_col_cell_str)

    return unique_tab_col_cells


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


def set_random_seed(seed=7):
    """
    Sets the random seed for the Python `random` module, the NumPy `random` module, and the Pandas `random` module.
    
    Args:
        seed (int): The seed value to use for the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)


def standardize_amb_unans_category(line):
    line['rawAmbiguousUnanswerableCategory'] = copy.deepcopy(line['ambiguousUnanswerableCategory'])
    category_lower = line['ambiguousUnanswerableCategory'].lower()
    if category_lower in ("Ambiguous_VALUES_across_Columns".lower(), "Ambiguous_VALUE_across_Column".lower(), "Ambiguous_VALUE_across_Columns".lower(), "Ambiguous_VALUES_across_Column".lower()):
        line['ambiguousUnanswerableCategory'] = "Ambiguous_WHERE_Column"
    elif line['ambiguousUnanswerableCategory'] == "Ambiguous_Filter_Term":
        line['ambiguousUnanswerableCategory'] = "Ambiguous_Filter_Criteria"
    elif category_lower in ("Nonexistent_Value".lower(), 'Nonexistent_Filter_Value'.lower(), "Nonexistent_Fliter_Value".lower()):
        line['ambiguousUnanswerableCategory'] = "Nonexistent_Filter_Value"
    elif category_lower in ("Ambiguous_VALUES_within_Column".lower()):
        line['ambiguousUnanswerableCategory'] = "Ambiguous_Values_Within_Column"
    return line


def sample_questions_by_database(lines, n_question_per_db=4):
    db2lines = {}
    for line in lines:
        dbid = line['db_id']
        if dbid not in db2lines:
            db2lines[dbid] = []
        db2lines[dbid].append(line)
    sampled_lines = []
    for db, db_lines in db2lines.items():
        if len(db_lines) < n_question_per_db:
            sampled_lines.extend(db_lines)
            continue
        sampled_lines.extend(random.sample(db_lines, n_question_per_db))
    return sampled_lines


def is_numeric(obj):
    try:
        float(obj)
        return True
    except (ValueError, TypeError):
        return False


class BinaryClassificationFilter(object):
    def __init__(self, binary_classification_fp="str", classification_key="binaryClassificationResult___claude-3-sonnet___lexicalAndOracle"):
        lines = read_jsonl_file(binary_classification_fp)
        self._key_to_result = {}
        for binary in lines:
            binary = standardize_amb_unans_category(binary)
            binary_pred = binary[classification_key].get("parsed", "")
            flag = binary_pred.lower().strip() == binary['ambiguousUnanswerableCategory'].lower().strip()
            self._key_to_result[self._get_line_key(binary)] = flag

    def is_valid_line(self, line, answerable_always_true=False):
        nline = copy.deepcopy(line)
        nline = standardize_amb_unans_category(nline)
        if answerable_always_true == True and line['ambiguousUnanswerableCategory'] == "answerable":
            return True
        line_key = self._get_line_key(nline)
        if line_key in self._key_to_result:
            return self._key_to_result[line_key]
        else:
            logger.error(f"Unknown line with key: {line_key}")
            raise Exception(f"Unknown line within the binary classification result.")
            # return False

    @staticmethod
    def _get_line_key(line):
        question = line['question']
        db_id = line['db_id']
        category = line['ambiguousUnanswerableCategory']
        schema_mod = line['schemaModification']
        if schema_mod:
            schema_mod_str = json.dumps(schema_mod)
        else:
            schema_mod_str = ""
        key = f"{question}___{db_id}___{category}___{schema_mod_str}"
        return key
