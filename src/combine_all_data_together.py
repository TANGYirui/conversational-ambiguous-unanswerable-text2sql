#%%
from utils import (
    read_jsonl_file,
    read_json_file,
    write_jsonl,
    format_schema_to_markdown,
    set_random_seed,
    get_table_alias_to_name_map_from_sql,
    parse_for_where,
    get_select_table_column_info,
)
from custom_sql_engine import (
    DbWithModification,
    MISSING_CELL_VALUE,
)
from simple_cache import cache_results

import os
from typing import Dict, Tuple, List
from loguru import logger
from tqdm import tqdm
import copy
import random
import typer
import glob

CACHE_DIR = os.path.join(os.path.dirname(__file__), "__cache__")  # use current dir as a cache dire
os.makedirs(CACHE_DIR, exist_ok=True)
IGNORE_CACHE = False


set_random_seed(9)


# %%
# 
# 1. get lexical retrieved cell values
# 2. get oracle cell values together with lexically retrieved cell values
# 3. (optional) get embedding based cell values + lexical retrieved cell values
# retrievedCellValues
#   lexicalOnly
#   lexicalAndOracle
#   lexicalAndSemantic
#   (stretch) lexicalAndOracleFromSqlTables


# @cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def sort_cell_value_tables_based_on_table_from_sql(tab_col_cell_dict: Dict[str, Dict[str, List]], tables_in_sql: List[str], position="bottom"):
    """
    position can be either:
        bottom: tables mentioned in the SQL are at the bottom of the linearization
        top: tables mentioned in the SQL are at the top of the linearization
    """
    if not tables_in_sql:
        return tab_col_cell_dict
    other_tables = [tab for tab in tab_col_cell_dict if tab not in tables_in_sql]
    
    if position == "top":
        tab_order = tables_in_sql + other_tables
    elif position == "bottom":
        tab_order = other_tables + tables_in_sql
    else:
        raise Exception(f"Unknown position specified: {position}. The only allowed position options are: `bottom` or `top`")
    sorted_tab_col_cell_dict = {}
    for tab in tab_order:
        sorted_tab_col_cell_dict[tab] = tab_col_cell_dict[tab]
    return sorted_tab_col_cell_dict


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


def standardize_amb_unans_category(line):
    if line['ambiguousUnanswerableCategory'].lower() in ("Ambiguous_VALUES_across_Columns".lower(), "Ambiguous_VALUE_across_Column".lower(), "Ambiguous_VALUE_across_Columns".lower(), "Ambiguous_VALUES_across_Column".lower()):
        line['rawAmbiguousUnanswerableCategory'] = copy.deepcopy(line['ambiguousUnanswerableCategory'])
        line['ambiguousUnanswerableCategory'] = "Ambiguous_WHERE_Column"
    elif line['ambiguousUnanswerableCategory'] == "Ambiguous_Filter_Term":
        line['rawAmbiguousUnanswerableCategory'] = copy.deepcopy(line['ambiguousUnanswerableCategory'])
        line['ambiguousUnanswerableCategory'] = "Ambiguous_Filter_Criteria"
    elif line['ambiguousUnanswerableCategory'] == ("Nonexistent_Value", 'Nonexistent_Filter_Value'):
        line['rawAmbiguousUnanswerableCategory'] = copy.deepcopy(line['ambiguousUnanswerableCategory'])
        line['ambiguousUnanswerableCategory'] = "Nonexistent_Filter_Value"
    return line


def append_cell_values_from_sql_to_tab_col_cell_dict(tab_col_cell_dict: Dict[str, Dict[str, List]], oracle_cells_from_sql: List[Dict[str, str]]):
    tab_col_cell_dict = copy.deepcopy(tab_col_cell_dict)
    for single_tab_col_cell in oracle_cells_from_sql:
        tab = single_tab_col_cell['table']
        col = single_tab_col_cell['column']
        cell = single_tab_col_cell['value']
        if cell:
            logger.info(f"Appended cell `{cell}` to cell values list for table `{tab}` and column `{col}`")
            tab_col_cells_from_db = tab_col_cell_dict[tab][col]
            combined_cells = [cell]
            for db_cell in tab_col_cells_from_db:
                if db_cell not in combined_cells:
                    combined_cells.append(db_cell)
            tab_col_cell_dict[tab][col] = combined_cells
    return tab_col_cell_dict


def get_tables_and_cells_mentioned_from_conversation(final_conversation, online_db, tab_col_cell_dict: Dict[str, Dict[str, List]], original_sql_query: str):
    tables_mentioned_in_sql = []
    cells_mentioned_in_sql = []
    for turn in final_conversation:
        if "DB EXPERT" in turn:
            sql_query = turn['DB EXPERT']
            try:
                alias2table = get_table_alias_to_name_map_from_sql(sql_query=sql_query)
                unique_tables = list(alias2table.values()) + list(alias2table.keys())
                tables_mentioned_in_sql.extend(unique_tables)
                where_cnds = parse_for_where(sql_query=sql_query)
                cells_mentioned_in_sql.extend(where_cnds)
            except Exception as ex:  # noqa:
                continue
    alias2table = get_table_alias_to_name_map_from_sql(sql_query=original_sql_query)
    unique_tables = list(alias2table.values()) + list(alias2table.keys())
    tables_mentioned_in_sql.extend(unique_tables)

    grounded_tables_mentioned_in_sql = []
    for tab in tables_mentioned_in_sql:
        grounded_tab = online_db.get_grounded_table(table=tab)
        if grounded_tab and grounded_tab not in grounded_tables_mentioned_in_sql:
            grounded_tables_mentioned_in_sql.append(grounded_tab)
    logger.info(f"Grouned tables: {grounded_tables_mentioned_in_sql}")

    grounded_cells_mentioned_in_sql = []
    for cell in cells_mentioned_in_sql:
        grounded_tab_col = online_db.get_grounded_table_column(table=cell['table'], column=cell['column'])
        tab_col_cells = tab_col_cell_dict.get(grounded_tab_col['table'], {}).get(grounded_tab_col['column'], [])
        tab_col_cells_lower = [str(cell).lower() for cell in tab_col_cells if cell not in (None, MISSING_CELL_VALUE)]
        if cell.get("operator", None) in ("=", "<>") and str(cell['value']).lower() in tab_col_cells_lower:
            grounded_cells_mentioned_in_sql.append(
                {"table": grounded_tab_col['table'], "column": grounded_tab_col['column'], "value": cell['value'], "operator": cell['operator']}
            )
            logger.info(f"Find grounded cell: {cell}")
    logger.info(f"Grounded cells: {grounded_cells_mentioned_in_sql}")

    return grounded_tables_mentioned_in_sql, grounded_cells_mentioned_in_sql


# %%
# 1. two way classification
# 2. 9-way classifcation (adding answerable data)
# 3. use LITELLM to simplify the experiments
# 4. (stretch) use agentic way to perform the classification and cell value retrieval


def run_main_combine_all_data(
    answerable_fp: str = "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/combined_data_all/spider/dev.json",
    output_dir: str = "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output_dev_0805",
    input_data_dir: str = "",
    n2sample: int = 0,
    shuffle_across_categories: int = 0,
):

    # # amb/unans data fp
    # If input_data_dir is provided, use it to find the generated category files
    # Otherwise use the hardcoded paths for backward compatibility
    if input_data_dir:
        # input_data_dir should point to the "dev" subdirectory containing the category files
        fp_list = [
            os.path.join(input_data_dir, "Ambiguous_SELECT_Column.jsonl"),
            os.path.join(input_data_dir, "Ambiguous_VALUES_across_Columns.jsonl"),
            os.path.join(input_data_dir, "Ambiguous_VALUES_within_Column.jsonl"),
            os.path.join(input_data_dir, "Ambiguous_Filter_Term.jsonl"),
            os.path.join(input_data_dir, "Nonexistent_Value.jsonl"),
            os.path.join(input_data_dir, "Nonexistent_WHERE_Column.jsonl"),
            os.path.join(input_data_dir, "Unsupported_Join.jsonl"),
            os.path.join(input_data_dir, "Nonexistent_SELECT_Column.jsonl"),
        ]
    else:
        fp_list = [
            "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output-20251228_151020/dev/Ambiguous_SELECT_Column.jsonl",
            "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output-20251228_151020/dev/Ambiguous_VALUES_across_Columns.jsonl",
            "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output-20251228_151020/dev/Ambiguous_VALUES_within_Column.jsonl",
            "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output-20251228_151020/dev/Ambiguous_Filter_Term.jsonl",
            "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output-20251228_151020/dev/Nonexistent_Value.jsonl",
            "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output-20251228_151020/dev/Nonexistent_WHERE_Column.jsonl",
            "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output-20251228_151020/dev/Unsupported_Join.jsonl",
            "/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output-20251228_151020/dev/Nonexistent_SELECT_Column.jsonl",
        ]

    # fp_list = list(glob.glob("/home/ubuntu/workspace/amb_unans_text2sql/ambi-unans-text-to-sql/.vscode/output_0728/train/*.jsonl"))
    if n2sample:
        out_fp = os.path.join(output_dir, f"amb_unans_ans_combined_sampled_{n2sample}_per_category{os.path.basename(answerable_fp)}")
    else:
        out_fp = os.path.join(output_dir, f"amb_unans_ans_combined_{os.path.basename(answerable_fp)}")
    if out_fp.endswith(".json"):
        out_fp += "l"

    spider_root_dir = os.path.dirname(answerable_fp)

    all_data = []
    for fp in fp_list:
        lines = read_jsonl_file(fp)
        for line in lines:
            line['finalConversation'] = line['ambiguousUnanswerableConversation']['rephrased_explanation_selected_followup_sql_complete_conversation']
        top_line = lines[0]
        logger.info(f"File path and category example:\nFilepath: {fp}\nCategory: {top_line['ambiguousUnanswerableCategory']}\nNumber of questions: {len(lines)}")
        if n2sample:
            if n2sample > len(lines):
                n2sample = len(lines)
            lines = random.sample(lines, k=n2sample)

        # HACK: newer unsupported join data will already be fitlered for this
        if "unsupported_join" in line['ambiguousUnanswerableCategory'].lower():
            lines = [
                line for line in lines
                if not is_removed_join_column_also_select_where_clause(line)
            ]

        # standarize the question category
        for line in lines:
            line = standardize_amb_unans_category(line)

        all_data.extend(lines)

    # load answerable data
    lines = read_json_file(answerable_fp)
    for line in lines:
        line['ambiguousUnanswerableCategory'] = "answerable"
        line['schemaModification'] = {}
        line['finalConversation'] = [
            {"USER": line['question']},
            {"DB EXPERT": line['query']},
        ]

    # all_data = []
    top_line = lines[0]
    logger.info(f"File path and category example:\nFilepath: {answerable_fp}\nCategory: {top_line['ambiguousUnanswerableCategory']}\nNumber of questions: {len(lines)}")
    if n2sample:
        if n2sample > len(lines):
            n2sample = len(lines)
        lines = random.sample(lines, k=n2sample)
    all_data.extend(lines)
    if shuffle_across_categories:
        random.shuffle(all_data)

    for line in tqdm(all_data):
        spider_database_dir = os.path.join(spider_root_dir, "database")
        db_id = line['db_id']
        initial_question = line['finalConversation'][0]['USER']  # initial question
        schema_modification = line.get("schemaModification", {})
        
        try:
            online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
        except Exception as ex:
            logger.error(f"Error loading database with modification: {ex}")
            continue

        # lexical_relevant_cells = get_lexically_related_cell_values(question=question, all_unique_cell_values=all_cell_values)
        lexical_relevant_cells = online_db.get_cell_values(only_unique_value=True, sort_order="alphabet", query=initial_question, threshold=0)
        # when threshold is 0, all cell values will be returned
        # all_cell_values = online_db.get_cell_values(only_unique_value=True, sort_order="alphabet")
        grounded_tables_mentioned_in_sql, grounded_cells_mentioned_in_sql = get_tables_and_cells_mentioned_from_conversation(
            final_conversation=line['finalConversation'],
            online_db=online_db,
            tab_col_cell_dict=copy.deepcopy(lexical_relevant_cells),
            original_sql_query=line['query']
        )

        # get lexically related cell values
        retrievedCellValues = {
            "lexicalOnly": None,
            "lexicalAndOracle": None,  # with used tables at the top OR bottom
            "lexicalAndSemantic": None,
            # "lexicalOracleFromSqlTables": None,
        }

        retrievedCellValues['lexicalOnly'] = copy.deepcopy(lexical_relevant_cells)
        if line['ambiguousUnanswerableCategory'] == "Ambiguous_VALUES_across_Columns":
            retrievedCellValues['lexicalAndOracle'] = online_db.get_oracle_cell_values_for_amb_values_across_column(
                only_unique_value=True, sort_order="alphabet", query=initial_question
            )
        elif line['ambiguousUnanswerableCategory'] == "Ambiguous_VALUES_within_Column":
            retrievedCellValues['lexicalAndOracle'] = online_db.get_oracle_cell_values_for_amb_values_within_column(
                only_unique_value=True,
                sort_order="alphabet",
                query=initial_question,
            )
        else:
            retrievedCellValues['lexicalAndOracle'] = append_cell_values_from_sql_to_tab_col_cell_dict(
                tab_col_cell_dict=lexical_relevant_cells,
                oracle_cells_from_sql=grounded_cells_mentioned_in_sql,
            )

        retrievedCellValues['lexicalAndOracle'] = sort_cell_value_tables_based_on_table_from_sql(
            retrievedCellValues['lexicalAndOracle'],
            tables_in_sql=grounded_tables_mentioned_in_sql,
        )

        line['retrievedCellValues'] = retrievedCellValues
        logger.info(f"Cell retrieve results:\nquestion: {initial_question}\ncategory: {line['ambiguousUnanswerableCategory']}\ncells:\n{format_schema_to_markdown(retrievedCellValues['lexicalAndOracle'])}\nschema modification:\n{line['schemaModification']}\n{'===' * 15}")

    fp_list_str = '\n'.join(fp_list)
    logger.info(f"Write data to: {out_fp}. All files combined:\n{fp_list_str}")
    write_jsonl(all_data, out_fp)


if __name__ == "__main__":
    typer.run(run_main_combine_all_data)
