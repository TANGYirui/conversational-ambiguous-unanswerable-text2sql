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


from custom_sql_engine import (
    DbWithModification,
    MISSING_CELL_VALUE,
)
from utils import (
    set_random_seed,
    format_schema_to_markdown,
    create_simple_message,
    read_jsonl_file,
    sample_questions_by_database,
    write_jsonl,
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


CURRENT_DIR = os.path.dirname(__file__)

set_random_seed(7)


def fix_conversation_copying_error(line, spider_root_dir):
    if line['ambiguousUnanswerableCategory'] == "answerable":
        return line

    spider_database_dir = os.path.join(spider_root_dir, "database")
    schema_modification = line['schemaModification']
    db_id = line['db_id']

    try:
        online_db = DbWithModification(db_id_name=db_id, database_main_dir=spider_database_dir, schema_modification=schema_modification)
    except Exception as ex:
        logger.error(f"Error loading database: {ex}")

    if line['finalConversation'][2]['USER'] == "FILL-IN-YOUR-RESPONSE-HERE" and line['finalConversation'][3]['DB EXPERT'] == "FILL-IN-YOUR-SQL-RESPONSE-HERE":
        ranked_pairs = line['ambiguousUnanswerableConversation'].get('ranked_response_with_followup_sql_parsed', [])
        if isinstance(ranked_pairs, dict):
            ranked_pairs = [ranked_pairs]
        original_paris = line['ambiguousUnanswerableConversation'].get('output_response_with_followup_and_sql_parsed', [])
        if isinstance(original_paris, dict):
            original_paris = [original_paris]
        followup_question_sql = ranked_pairs + original_paris
        for pair in followup_question_sql:
            if not pair:
                continue
            try:
                sql = pair['DB EXPERT']
                online_db.run_sql(sql)
                line['finalConversation'][2]['USER'] = pair['USER']
                line['finalConversation'][3]['DB EXPERT'] = pair['DB EXPERT']
                logger.info(f"Selected followup question & SQL pair: {pair}")
                return line
            except Exception as ex:
                logger.info(f"Pair: {pair}")
                logger.info(f"Failed to execute the SQL: {sql}\nschema modification: {schema_modification}\ncategory: {line['ambiguousUnanswerableCategory']}")
                pass

        # logger.info(f"Problematic line:\n{json.dumps(line, indent=2)}")
        updated_followup_and_sql = line['ambiguousUnanswerableConversation']['ranked_response_with_followup_sql_parsed']
        # sometimes if there are to many candidates to rerank, the reranking may fail
        if not updated_followup_and_sql:
            updated_followup_and_sql = line['ambiguousUnanswerableConversation']['output_response_with_followup_and_sql_parsed'][0]
        else:
            updated_followup_and_sql = updated_followup_and_sql[0]
            if "<specific_document_id>" in updated_followup_and_sql or "[DOCUMENT_ID]" in updated_followup_and_sql:
                updated_followup_and_sql = line['ambiguousUnanswerableConversation']['output_response_with_followup_and_sql_parsed'][0]
        line['finalConversation'][2]['USER'] = updated_followup_and_sql['USER']
        line['finalConversation'][3]['DB EXPERT'] = updated_followup_and_sql['DB EXPERT']
        return line
    return line


def contextualize_followup_question_single_line(
    line,
    spider_root_dir: str,
    system_prompt: str,
    few_shots: List[Dict],
    litellm_router=None,
):
    spider_database_dir = os.path.join(spider_root_dir, "database")
    schema_modification = line['schemaModification']
    db_id = line['db_id']

    # fix some errors in copying conversation during formatting process
    line = fix_conversation_copying_error(line, spider_root_dir=spider_root_dir)

    final_conv = line['finalConversation']
    cell_values = line['retrievedCellValues']['lexicalAndOracle']
    schema_preview = format_schema_to_markdown(cell_values)

    if line['ambiguousUnanswerableCategory'] == "answerable":
        line['finalConversationWithConextualization'] = final_conv
        return line

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<conversation>\n\n{json.dumps(final_conv, indent=2)}\n\n</conversation>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user", message_type="litellm")]
    tmp_msgs = [{"role": "system", "content": system_prompt}] + tmp_msgs
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, model="gpt-4o-mini", router=litellm_router)
    # print(user_msg)
    # print("---" * 30)
    # print(response)
    try:
        result = extract_string_list_from_xml_tags(response, "result")[0].strip()
        final_conversation_with_followup = copy.deepcopy(final_conv)
        final_conversation_with_followup[2]['USER'] = result
    except Exception as ex:
        line_to_print = copy.deepcopy(line)
        line_to_print.pop("retrievedCellValues", None)
        logger.error(f"Error parsing result: {ex}.\nResponse: {response}\nLine data: {json.dumps(line_to_print, indent=2)}")
        # use original conversation as back up if no rephrase is generated
        # shall rarely happen
        final_conversation_with_followup = copy.deepcopy(final_conv)
    
    line['finalConversationWithConextualization'] = final_conversation_with_followup
    return line


def add_execution_explanation_single_line(
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

    final_conv = line['finalConversation']
    cell_values = line['retrievedCellValues']['lexicalAndOracle']
    schema_preview = format_schema_to_markdown(cell_values)

    try:
        execution_results = online_db.run_sql(
            final_conv[-1]['DB EXPERT']
        )
        execution_results = execution_results[:30]  # only show the first 30 rows of execution results
    except Exception as ex:
        logger.error(f"Unexpected SQL execution error: {ex}. \nSQL: {final_conv[-1]['DB EXPERT']}\nSchema Modification: {schema_modification}")
        # TODO: better handling of such unexpected execution errors
        # The SQL shall be working during SQL generation process.
        execution_results = []

    user_msg = f"<schema>\n\n{schema_preview}\n\n</schema>\n\n\n<conversation>\n\n{json.dumps(final_conv, indent=2)}\n\n</conversation>\n\n<execution_results>\n```\n{execution_results}\n```\n</execution_results>"
    tmp_msgs = few_shots + [create_simple_message(message=user_msg, role="user", message_type="litellm")]
    tmp_msgs = [{"role": "system", "content": system_prompt}] + tmp_msgs
    response = router_completion_with_ratelimit_retry(messages=tmp_msgs, model="gpt-4o-mini", router=litellm_router)
    # print(user_msg)
    # print("---" * 30)
    # print(response)

    final_conversation_with_followup = copy.deepcopy(line['finalConversationWithConextualization'])
    try:
        result = extract_string_list_from_xml_tags(response, "result")[0].strip()
        final_conversation_with_followup.append(
            {
                "DB EXPERT": result
            }
        )
    except Exception as ex:
        logger.error(f"Error parsing result: {ex}")
        # use original conversation as back up if no rephrase is generated
        # shall rarely happen
    
    line['finalConversationWithConextualizationAndExecutionExplanation'] = final_conversation_with_followup
    return line


def process_single_line(
    line,
    spider_root_dir,
    contextualize_system_prompt,
    contextualize_fewshots,
    execution_explanation_system_prompt,
    execution_explanation_fewshots,
):
    # Each worker creates its own router to avoid serialization overhead
    litellm_router = get_litellm_router()

    # rephrase the templated explanation
    line_with_contextualized_followup = contextualize_followup_question_single_line(
        line=line,
        spider_root_dir=spider_root_dir,
        system_prompt=contextualize_system_prompt,
        few_shots=contextualize_fewshots,
        litellm_router=litellm_router,
    )
    if line_with_contextualized_followup is None:
        return None

    # rephrase the templated explanation
    line_with_rephrased_explanation = add_execution_explanation_single_line(
        line=line_with_contextualized_followup,
        spider_root_dir=spider_root_dir,
        system_prompt=execution_explanation_system_prompt,
        few_shots=execution_explanation_fewshots,
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
    contextualize_fewshots_fn = "contextualize_fewshots_system.ipynb"
    contextualize_fewshots_path = os.path.join(CURRENT_DIR, "contextualize_execution_explanation_prompts", contextualize_fewshots_fn)
    contextualize_fewshots, contextualize_system_prompt = add_fewshots_from_path(path_str=contextualize_fewshots_path, extension=".ipynb")
    contextualize_fewshots = convert_claude_msg_list_to_litellm_msg_list(contextualize_fewshots)

    # having a critic model to ensure that the generated cell values are all valid
    execution_explanation_fn = "execution_explanation_fewshots_system.ipynb"
    execution_explanation_path = os.path.join(CURRENT_DIR, "contextualize_execution_explanation_prompts", execution_explanation_fn)
    execution_explanation_fewshots, execution_explanation_system_prompt = add_fewshots_from_path(path_str=execution_explanation_path, extension=".ipynb")
    execution_explanation_fewshots = convert_claude_msg_list_to_litellm_msg_list(execution_explanation_fewshots)


    nlines = []
    # Each worker creates its own router inside process_single_line
    with multiprocessing.Pool(processes=num_workers) as pool:
        nlines = list(tqdm(pool.imap(
            partial(
                process_single_line,
                spider_root_dir=spider_root_dir,
                contextualize_system_prompt=contextualize_system_prompt,
                contextualize_fewshots=contextualize_fewshots,
                execution_explanation_system_prompt=execution_explanation_system_prompt,
                execution_explanation_fewshots=execution_explanation_fewshots,
            ),
            lines
        ), total=len(lines)))


    return nlines


def filter_by_binary_classification(
    lines: List[Dict],
    binary_classification_lines: List[Dict],
    llm_model: str,
    cell_value_key: str,
) -> List[Dict]:
    """
    Filter lines based on binary classification results.

    Only keep examples where the classification matches the expected category
    (i.e., the model correctly classified the example as its labeled category).
    """
    # Create lookup dictionary by question_id or unique identifier
    # We'll use db_id + initial question as a key
    classification_lookup = {}
    for line in binary_classification_lines:
        key = (line['db_id'], line['finalConversation'][0]['USER'])
        result_key = f"binaryClassificationResult___{llm_model}___{cell_value_key}"
        if result_key in line:
            parsed_result = line[result_key].get('parsed')
            classification_lookup[key] = parsed_result

    # Filter original lines
    filtered_lines = []
    filtered_count = 0
    for line in lines:
        key = (line['db_id'], line['finalConversation'][0]['USER'])
        expected_category = line['ambiguousUnanswerableCategory']

        # If we have classification results, check if they match
        if key in classification_lookup:
            predicted_category = classification_lookup[key]

            # Keep if classification matches, or if it's answerable (we keep those)
            if expected_category == "answerable":
                filtered_lines.append(line)
            elif predicted_category == expected_category:
                filtered_lines.append(line)
            else:
                filtered_count += 1
                logger.debug(f"Filtered out: expected={expected_category}, predicted={predicted_category}")
        else:
            # No classification available, keep the line
            filtered_lines.append(line)

    logger.info(f"Binary classification filtering: kept {len(filtered_lines)}/{len(lines)} examples (filtered {filtered_count})")
    return filtered_lines


def main(
    spider_data_root_dir: str = typer.Option(
        '../../spider/dataset', help='The path to the Spider dataset'
    ),
    infp: str = typer.Option(
        "", help="The file path to the combined data files that contain all types of questions"
    ),
    binary_classification_fp: str = typer.Option(
        "", help="Optional: file path to the binary classification results for filtering to improve data quality"
    ),
    classification_llm_model: str = typer.Option(
        "claude-3-5-sonnet", help="LLM model used for classification (needed to parse classification results)"
    ),
    classification_cell_value_key: str = typer.Option(
        "lexicalAndOracle", help="Cell value key used for classification (needed to parse classification results)"
    ),
    n2sample: int = typer.Option(0, help='Sample the questions'),
    num_workers: int = typer.Option(7, help='Number of parallel workers for processing'),
):

    # # # extract questions with join and include metadata: ambiguousUnanswerableCategory & schemaModification
    # lines_of_interest = get_questions_with_select_clause(spider_data=split_all, spider_root_dir=spider_data_root_dir)
    lines = read_jsonl_file(infp)
    logger.info(f"Loaded {len(lines)} examples from {infp}")

    if n2sample:
        lines = sample_questions_by_database(lines=lines, n_question_per_db=n2sample)
        logger.info(f"Sampled down to {len(lines)} examples ({n2sample} per database)")

    # Apply binary classification filtering if provided
    if binary_classification_fp and os.path.exists(binary_classification_fp):
        logger.info(f"Loading binary classification results from: {binary_classification_fp}")
        binary_classification_lines = read_jsonl_file(binary_classification_fp)
        lines = filter_by_binary_classification(
            lines=lines,
            binary_classification_lines=binary_classification_lines,
            llm_model=classification_llm_model,
            cell_value_key=classification_cell_value_key,
        )
    elif binary_classification_fp:
        logger.warning(f"Binary classification file specified but not found: {binary_classification_fp}")

    nlines = process_lines_of_interest(
        lines=lines,
        spider_root_dir=spider_data_root_dir,
        num_workers=num_workers,
    )

    nlines = [line for line in nlines if line]
    output_jsonl_fp = infp + ".contextualize_and_execulation_explanation_v2.jsonl"
    logger.info(f"Output file: {output_jsonl_fp}")
    write_jsonl(nlines, output_jsonl_fp)
    logger.info(f"Wrote {len(nlines)} contextualized examples")
    

if __name__ == '__main__':
    # TODO: fixed the fewshots example with new templated ambiguosu explanation
    # TODO: add a filtering step to remove un-natural conversations
    typer.run(main)
