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
    load_spider_dev_train_data,
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


AMB_UNANS_CATEGORY = "Ambiguous_Filter_Term"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "__cache__")  # use current dir as a cache dire
os.makedirs(CACHE_DIR, exist_ok=True)
IGNORE_CACHE = False
CURRENT_DIR = os.path.dirname(__file__)

set_random_seed(7)


@cache_results(CACHE_DIR, ignore_cache=IGNORE_CACHE)
def identify_where_column_for_removal(line, spider_root_dir):
    # spider_database_dir = os.path.join(spider_root_dir, "database")
    # db_id = line['db_id']
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
        # online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=None)
        # tab_col_cells = online_db.get_cell_values(only_unique_value=True)
        # lexically_related_columns = get_lexically_similar_columns_from_schema(single_where_clause, tab_col_cells=tab_col_cells)
        line['ambiguousUnanswerableCategory'] = AMB_UNANS_CATEGORY
        line['schemaModification'] = {}
        line['vagueTermColumn'] = {
            "table": single_where_clause['table'], "column": single_where_clause['column'], "value": single_where_clause['value'],
        }
        return line
    else:
        if "select" in line['query'].lower() and "*" not in line['query']:
            logger.debug(f"No select clause: {line['query']}")
        # logger.debug(f"No where clause: {data['query']}")
        return None


@cache_results(cache_path=CACHE_DIR, ignore_cache=IGNORE_CACHE)
def generate_vague_filter_term_for_single_line(
    line, spider_root_dir, 
    system_prompt, few_shots, 
):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    db_id = line['db_id']
    column_of_interest = line.get("vagueTermColumn", {})

    try:
        # we show the complete schema to help model find better replacement columns that will be ambiguous
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification={})
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        return {}
    # generate templated unanswerable explanation
    schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<column>\n{json.dumps(column_of_interest)}\n</column>\n\n<question>\n{line['question']}\n</question>\n\n<sql>\n{line['query']}\n</sql>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
    # print(user_msg)
    # print("---" * 20)
    # print(response)
    # print("===" * 20)
    try:
        result = extract_string_list_from_xml_tags(response, "result")[0].strip()
        if not result:
            logger.error(f"Model predicted no vague term ambiguous rephrasing. Result:\n{json.dumps(result, indent=2)}")
            return None
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        result = {}

    conversation_list = [
        {
            "USER": result
        },
        {
            "DB EXPERT": "FILL-IN-YOUR-RESPONSE-HERE",
        },
        {
            "USER": line['question']
        },
        {
            "DB EXPERT": line['query']
        }
    ]

    line['ambiguousUnanswerableConversation'] = {
        "input_conversation_with_placeholder_to_fill": conversation_list,
    }

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
def rephrase_the_templated_explanation_single_line(
    line,
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict],
):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    schema_modification = line['schemaModification']
    column_of_interest = line.get("vagueTermColumn", {})
    column_of_interest.pop("value", None)
    db_id = line['db_id']

    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")
        # online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)

    schema_preview = format_schema_to_markdown(online_db.get_cell_values(only_unique_value=True))
    conversation_with_template = line['ambiguousUnanswerableConversation']['input_conversation_with_placeholder_to_fill']

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n<column>\n{json.dumps(column_of_interest)}\n</column>\n\n\n<conversation>\n\n{json.dumps(conversation_with_template, indent=2)}\n\n</conversation>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user")]
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, system=system_prompt)
    # print(user_msg)
    # print("---" * 30)
    # print(response)
    # print("===" * 30)
    try:
        result = extract_string_list_from_xml_tags(response, "result")[0].strip()
        final_conversation_with_followup = copy.deepcopy(conversation_with_template)
        final_conversation_with_followup[1]['DB EXPERT'] = result
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        final_conversation_with_followup = {}
        result = {}

    if not result:
        return None
    
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
):

    # follow-up question generation
    vague_term_question_few_shots_filename = "fewshots_vague_term_question_generation.ipynb"
    vague_term_few_shots_path = os.path.join(CURRENT_DIR, vague_term_question_few_shots_filename)
    vague_term_few_shots, vague_term_system_prompt = add_fewshots_from_path(path_str=vague_term_few_shots_path, extension=".ipynb")

    # rephrase the templated explanation
    rephrase_templated_explanation_few_shots_filename = "fewshots_examples_for_asking_for_clarification.ipynb"
    rephrase_templated_explanation_few_shots_path = os.path.join(CURRENT_DIR, rephrase_templated_explanation_few_shots_filename)
    rephrase_templated_explanation_few_shots, rephrase_templated_explanation_system_prompt = add_fewshots_from_path(path_str=rephrase_templated_explanation_few_shots_path, extension=".ipynb")

    nlines = []
    # lines = sample_questions_by_database(lines=lines, n_question_per_db=5)
    for line in tqdm(lines):
        # identify the select Clause for removal
        line_with_select_clause = identify_where_column_for_removal(line=line, spider_root_dir=spider_root_dir)
        if not line_with_select_clause:
            continue

        # generate replacement columns and modify the SQL accordingly
        line_with_vague_term_question = generate_vague_filter_term_for_single_line(
            line=line_with_select_clause,
            spider_root_dir=spider_root_dir,
            system_prompt=vague_term_system_prompt,
            few_shots=vague_term_few_shots, 
        )
        if not line_with_vague_term_question:
            continue

        # rephrase the templated explanation
        line_with_rephrased_explanation = rephrase_the_templated_explanation_single_line(
            line=line_with_vague_term_question,
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
