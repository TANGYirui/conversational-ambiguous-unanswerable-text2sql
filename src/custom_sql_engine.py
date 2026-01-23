import os
import shutil
import sqlite3
from typing import Dict, List
from loguru import logger
import copy
import random

from rapidfuzz import fuzz
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


MISSING_CELL_VALUE = "___MISSING_VALUE___"
INVALID_TABLE_NAMES = [
    "sqlite_sequence",
]

porter_stemmer = PorterStemmer()


def _get_sorted_lexically_related_cell_values_based_on_question(question, all_unique_cell_values, threshold=0, max_k=500000000000):
    # check whether a cell value is token within the question
    # if yes, add it, 
    # tokenize the question into list of tokens
    question_lower = question.lower()
    question_tokens = word_tokenize(question_lower)
    stemmed_question_tokens = [porter_stemmer.stem(word) for word in question_tokens]
    all_question_tokens_lower = set(question_tokens + stemmed_question_tokens)

    relevant_cell_values = {}
    relevant_cell_values_above_threshold = {}
    for tab in all_unique_cell_values:
        col_cells = all_unique_cell_values[tab]
        if tab not in relevant_cell_values:
            relevant_cell_values[tab] = {}
            relevant_cell_values_above_threshold[tab] = {}

        for col in col_cells:
            if col not in relevant_cell_values[tab]:
                relevant_cell_values[tab][col] = []
                relevant_cell_values_above_threshold[tab][col] = []

            cells = col_cells[col]
            for cell in cells:
                if cell in (None, MISSING_CELL_VALUE):
                    continue
                # 1. check whether the cell values is within the question
                cell_str = str(cell)
                cell_lower = cell_str.lower()
                if cell_lower in all_question_tokens_lower:
                    relevant_cell_values[tab][col].append(
                        {"similarity": 100, "cell": cell}
                    )
                    continue
                # 2. check whether the stemmed cell values is within the question
                stemmed_cell_lower = porter_stemmer.stem(cell_lower)
                if stemmed_cell_lower in all_question_tokens_lower:
                    relevant_cell_values[tab][col].append(
                        {"similarity": 99.5, "cell": cell}
                    )
                    continue
                # 3. calculate the lexical similarity between the cell and the question
                similarity = fuzz.partial_ratio(cell_lower, question_lower)
                relevant_cell_values[tab][col].append(
                    {"similarity": similarity, "cell": cell}
                )
            # sort all the cell values based on the similarity score
            cells_with_similarity_score = relevant_cell_values[tab][col]
            sorted_cells = sorted(cells_with_similarity_score, key=lambda x: x['similarity'], reverse=True)
            relevant_cell_values[tab][col] = sorted_cells
            relevant_cell_values_above_threshold[tab][col] = [
                cell_score['cell'] for cell_score in sorted_cells
                if cell_score['similarity'] >= threshold
            ][:max_k]

    return {
        "allCellWithSimilarityScore": relevant_cell_values,
        "cellOnlyAboveThreshold": relevant_cell_values_above_threshold,
    }


def get_table_names(conn):
    """
    Returns a list of table names in a SQLite3 database.

    Args:
        conn (sqlite3.Connection): A SQLite3 database connection object.

    Returns:
        list: A list of table names in the database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # sometimes the database contain a table called `sqlite_sequence`, it is not useful but will create exception
    tables = [table[0] for table in cursor.fetchall() if table[0].lower() not in INVALID_TABLE_NAMES]
    cursor.close()
    return tables


def create_in_memory_db(db_path_or_conn):
    """
    Creates an in-memory SQLite3 database from an existing disk-based SQLite3 database.

    Args:
        db_path (str): The path to the disk-based SQLite3 database file.

    Returns:
        sqlite3.Connection: An in-memory SQLite3 database connection.
    """
    # Connect to the disk-based SQLite3 database
    if isinstance(db_path_or_conn, sqlite3.Connection):
        disk_conn = db_path_or_conn
    elif isinstance(db_path_or_conn, str):
        disk_conn = sqlite3.connect(db_path_or_conn)
    else:
        raise Exception("Invalid database connection")

    # fix encoding errors when copying data from original table to table in memory
    # https://stackoverflow.com/a/58891189
    disk_conn.text_factory = lambda b: b.decode(errors='ignore')
    disk_cursor = disk_conn.cursor()

    # Create an in-memory SQLite3 database
    mem_conn = sqlite3.connect(":memory:")
    mem_cursor = mem_conn.cursor()

    # Get the schema of the disk-based database
    disk_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    tables = disk_cursor.fetchall()

    # get the table names from disk database
    table_names = get_table_names(disk_conn)

    # Create tables in the in-memory database
    for table in tables:
        if "sqlite_sequence" not in table[0].lower():
        # if table.lower() not in INVALID_TABLE_NAMES:
            mem_cursor.execute(table[0])
        else:
            pass
            # logger.warning(f"Dropped invalid table: {table}")

    # Copy data from the disk-based database to the in-memory database
    for table_name in table_names:
        disk_cursor.execute(f"SELECT * FROM {table_name}")
        rows = disk_cursor.fetchall()
        if rows:
            # logger.info(f"Copied table: {table_name}")
            mem_cursor.executemany(f"INSERT INTO {table_name} VALUES ({','.join('?' * len(rows[0]))})", rows)
        else:
            # logger.warning(f"Table {table_name} is empty")
            pass

    # Commit changes to the in-memory database
    mem_conn.commit()

    # TODO: compare that the memory database and disk database are the same

    # Close the disk-based database connection
    disk_conn.close()

    return mem_conn


def add_column_and_insert_values(conn, table_name, column_name, column_type, values):
    """
    Add a new column to an existing SQLite table and insert values into the new column.

    Args:
        conn (sqlite3.Connection): The SQLite database connection.
        table_name (str): The name of the table to modify.
        column_name (str): The name of the new column to add.
        column_type (str): The data type of the new column (e.g., TEXT, INTEGER, REAL).
        values (list): A list of values to insert into the new column.

    Returns:
        None
    """
    cursor = conn.cursor()

    # Add the new column
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    except sqlite3.OperationalError as e:
        logger.error(f"Error adding column: {e}")
        return

    # Insert values into the new column
    update_query = f"UPDATE {table_name} SET {column_name} = ? WHERE rowid = ?"
    for row_id, value in enumerate(values, start=1):
        try:
            cursor.execute(update_query, (value, row_id))
        except sqlite3.Error as e:
            logger.error(f"Error inserting value: {e}")
    cursor.close()
    conn.commit()


def get_schema_modification_without_long_info(schema_modification, remove_related_info=True):
    schema_mod_copy = copy.deepcopy(schema_modification)
    schema_mod_copy.pop("LLM_Based_Alternative_Columns_To_Remove", None)
    if "addColumn" in schema_mod_copy:
        for item in schema_mod_copy['addColumn']:
            item.pop("value", None)
    if remove_related_info:
        schema_mod_copy.pop("removeColumnLexicallyRelated", None)
        schema_mod_copy.pop("removeColumnSemanticallyRelated", None)
        schema_mod_copy.pop("removeCellLexicallyRelated", None)
    return schema_mod_copy


class DbWithModification:
    # TODO: properly handle the foreign key relationship
    def __init__(self, db_id_name, database_main_dir, schema_modification):
        self._MISSING_VALUE = MISSING_CELL_VALUE
        self.database_main_dir = database_main_dir
        self.database_dir = os.path.join(self.database_main_dir, db_id_name)
        sqlite_file = os.path.join(self.database_dir, db_id_name + '.sqlite')
        self.in_memory_conn = create_in_memory_db(sqlite_file)
        # modify the database based on schema modification
        self._removed_tab_cols = []
        self._removed_tab_col_cells = []
        self._schema_modification = schema_modification
        if schema_modification:
            # handling column deletion
            if "removeColumn" in schema_modification:
                removed_columns = schema_modification["removeColumn"]
                for tab_col in removed_columns:
                    tab, col = tab_col['table'], tab_col['column']
                    status = self.delete_column_from_table(table_name=tab, column_name=col)
                    if status:
                        self._removed_tab_cols.append({"table": tab, "column": col})
            if "removeColumnLexicallyRelated" in schema_modification:
                removed_columns = schema_modification["removeColumnLexicallyRelated"]
                for tab_col in removed_columns:
                    tab, col = tab_col['table'], tab_col['column']
                    status = self.delete_column_from_table(table_name=tab, column_name=col)
                    if status:
                        self._removed_tab_cols.append({"table": tab, "column": col})
            if "removeColumnSemanticallyRelated" in schema_modification:
                removed_columns = schema_modification["removeColumnSemanticallyRelated"]
                for tab_col in removed_columns:
                    tab, col = tab_col['table'], tab_col['column']
                    status = self.delete_column_from_table(table_name=tab, column_name=col)
                    if status:
                        self._removed_tab_cols.append({"table": tab, "column": col})
            if "removeCell" in schema_modification:
                removed_cells = schema_modification["removeCell"]
                for tab_col_cell in removed_cells:
                    tab, col, cell = tab_col_cell['table'], tab_col_cell['column'], tab_col_cell['value']
                    status = self.delete_cell_from_table(tab=tab, col=col, cell=cell)
                    if status:
                        self._removed_tab_col_cells.append({"table": tab, "column": col, "value": cell})
            if "removeCellLexicallyRelated" in schema_modification:
                removed_cells = schema_modification["removeCellLexicallyRelated"]
                for tab_col_cell in removed_cells:
                    tab, col, cell = tab_col_cell['table'], tab_col_cell['column'], tab_col_cell['value']
                    status = self.delete_cell_from_table(tab=tab, col=col, cell=cell)
                    if status:
                        self._removed_tab_col_cells.append({"table": tab, "column": col, "value": cell})
            # raise Exception('Schema modification not implemented yet')
            if "addColumn" in schema_modification:
                added_columns = schema_modification["addColumn"]
                for tab_col_cell in added_columns:
                    tab, col, col_type = tab_col_cell['table'], tab_col_cell['column'], tab_col_cell['type']
                    cell_values = tab_col_cell['value']
                    self.add_column_to_table(table_name=tab, column_name=col, column_type=col_type, values=cell_values)
            # we add cell to locations where value is missing (either missing in original database or deleted in previous removeCell stage)
            if "addCell" in schema_modification:
                replaced_cells = schema_modification['addCell']
                # get all cell values from the column
                # find all cell values that are NULL or MISSING_VALUE
                # if there are less than two such cell values, insert MISSING_VALUE to the end
                # set these cell values with the replacement cell values one by one
                self.add_cell_to_table_column(replaced_cells)
                
    @property
    def removedColumns(self):
        return self._removed_tab_cols

    def run_sql(self, sql):
        cursor = self.in_memory_conn.cursor()
        result = cursor.execute(sql)
        result_all = result.fetchall()
        cursor.close()
        return result_all

    def get_oracle_cell_values_for_amb_values_across_column(self, only_unique_value=True, include_column_type=False, ignore_table_column_casing=False, sort_order="default", query=None):
        # only used in certain ORACLE/human testing cases
        # only useful for ambiguous values across columns data generation
        # if we delete a column and replace it with two replacement columns and the removed column contains cell values,
        # we want to make sure these cell values are in the replacment columns in oracle setting
        cell_values = self.get_cell_values(only_unique_value=only_unique_value, include_column_type=include_column_type, ignore_table_column_casing=ignore_table_column_casing, sort_order=sort_order, query=query)
        remove_col_cell = self._schema_modification['removeColumn'][0]
        for tab_col in self._schema_modification['addColumn']:
            grounded_tab_col = self.get_grounded_table_column(table=tab_col['table'], column=tab_col['column'])
            tab, col = grounded_tab_col['table'], grounded_tab_col['column']
            cell_values[tab][col] = [remove_col_cell['value']] + [cell for cell in cell_values[tab][col] if cell != remove_col_cell['value']]
        return cell_values

    def get_oracle_cell_values_for_amb_values_within_column(self, only_unique_value=True, include_column_type=False, ignore_table_column_casing=False, sort_order="default", query=None):
        # only used in certain ORACLE/human testing cases
        # only useful for ambiguous values within columns data generation
        # if we delete one cell value and replace it with other cell values, we want to ensure the newly added cell values appear at the begging of the cell value list
        cell_values = self.get_cell_values(only_unique_value=only_unique_value, include_column_type=include_column_type, ignore_table_column_casing=ignore_table_column_casing, sort_order=sort_order, query=query)
        # remove_col_cell = self._schema_modification['removeCell'][0]
        added_cells = self._schema_modification['addCell']
        for add_cell in added_cells:
            grounded_tab_col = self.get_grounded_table_column(table=add_cell['table'], column=add_cell['column'])
            tab, col = grounded_tab_col['table'], grounded_tab_col['column']
            cell_values[tab][col] = [add_cell['value']] + [cell for cell in cell_values[tab][col] if cell != add_cell['value']]
        return cell_values

    def get_cell_values(self, only_unique_value=True, include_column_type=False, ignore_table_column_casing=False, sort_order="default", query=None, threshold=0):
        """Get cell values from the schema.
        sort_order: default | alphabet | random  --> determine how the tables and columns are ordered, does not affect the order of retrieved cell values
            default: use the order from sqlite default output, not other handling
            alphabet: sort tables from a to z, and columns from a to z
            random: randomly shuffled table and column order
        query:
            if provided, the cell values will be ordered based on the fuzzy matched lexical similarity between each cell value and the question similarity
            if not provided, does not sort cell values based on query
        """
        table_col_cells = {}
        tables = get_table_names(self.in_memory_conn)

        if include_column_type and sort_order != "default":
            raise Exception("Only `default` sort order is supported when column type is included.")
        for tab in tables:
            if only_unique_value:
                table_col_cells[tab] = self.get_distinct_values_from_table(self.in_memory_conn, tab, include_column_type=include_column_type)
            else:
                table_col_cells[tab] = self.get_all_values_from_table(self.in_memory_conn, tab, include_column_type=include_column_type)
        if ignore_table_column_casing:
            result = {}
            for tab, col_cells in table_col_cells.items():
                result[tab.lower().strip()] = {
                    key.lower().strip(): val for key, val in col_cells.items()
                }
            table_col_cells = result
            # return result
        # else:
        if sort_order == "default":  # no sorting
            sorted_table_col_cells = table_col_cells
        elif sort_order in ("random", "alphabet"):
            keys = list(table_col_cells.keys())
            if sort_order == "random":
                random.shuffle(keys)
            elif sort_order == "alphabet":
                keys.sort()
            sorted_table_col_cells = {}
            for k in keys:
                sub_keys = list(table_col_cells[k].keys())
                if sort_order == "random":
                    random.shuffle(sub_keys)
                elif sort_order == "alphabet":
                    sub_keys.sort()
                sorted_table_col_cells[k] = {
                    subk: table_col_cells[k][subk] for subk in sub_keys
                }
        else:
            raise Exception(f"Unknown sort order for the retrieved cell values: {sort_order}")

        # if query is not None
        if query:
            query_sorted_cell_values = _get_sorted_lexically_related_cell_values_based_on_question(
                question=query,
                all_unique_cell_values=sorted_table_col_cells,
                threshold=threshold,
            )
            sorted_table_col_cells = query_sorted_cell_values['cellOnlyAboveThreshold']

        return sorted_table_col_cells

    @staticmethod
    def get_distinct_values_from_table(conn, table_name, include_column_type=False, include_all_cell_values=False):
        """
        Returns a dictionary containing distinct values for each column in a table.

        Args:
            conn (sqlite3.Connection): A SQLite3 database connection object.
            table_name (str): The name of the table to extract distinct values from.

        Returns:
            dict: A dictionary where the keys are column names and the values are lists
                of distinct values in those columns.
        """
        cursor = conn.cursor()

        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_with_metadata = [col for col in cursor.fetchall()]
        columns = [col[1] for col in columns_with_metadata]
        column_types = [col[2] for col in columns_with_metadata]

        distinct_values = {}

        # Get distinct values for each column
        for original_column, col_type in zip(columns, column_types):
            # for unrecognized token, this may raise exceptions
            if '"' not in original_column:
                column = f'"{original_column}"'
            if '"' not in table_name:
                table_name = f'"{table_name}"'
            cell_retrieval_cmd = f"SELECT DISTINCT {column} FROM {table_name}"
            if include_all_cell_values:
                cell_retrieval_cmd = f"SELECT {column} FROM {table_name}"
            cursor.execute(cell_retrieval_cmd)

            cell_values = [row[0] for row in cursor.fetchall()]
            if include_column_type:
                distinct_values[original_column] = {
                    "cell_values": cell_values,
                    "type": col_type,
                }
            else:
                distinct_values[original_column] = cell_values

        cursor.close()
        return distinct_values

    def get_all_values_from_table(self, conn, table_name, include_column_type=False):
        """
        Returns a dictionary containing distinct values for each column in a table.

        Args:
            conn (sqlite3.Connection): A SQLite3 database connection object.
            table_name (str): The name of the table to extract distinct values from.

        Returns:
            dict: A dictionary where the keys are column names and the values are lists
                of distinct values in those columns.
        """
        distinct_values = self.get_distinct_values_from_table(
            conn=conn,
            table_name=table_name,
            include_column_type=include_column_type,
            include_all_cell_values=True,
        )
        return distinct_values

    def close(self):
        self.in_memory_conn.close()

    def get_schema(self):
        cursor = self.in_memory_conn.cursor()
        tables = get_table_names(self.in_memory_conn)
        table2schema = {}
        # Get column names
        for table_name in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_with_metadata = [col for col in cursor.fetchall()]
            schema = {
                "columns": [col[1] for col in columns_with_metadata],
                "columnTypes": [col[2] for col in columns_with_metadata],
            }

            # Get foreign key constraints for the table
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = cursor.fetchall()
            schema['<FOREIGN_KEYS_RELATIONSHIP>'] = foreign_keys

            # primary key relationship
            primary_key_str = self.get_primary_key_constraints_after_deleting_tab_col(table_name=table_name, column_name=None)
            if primary_key_str:
                schema['<PRIMARY_KEY_RELATIONSHIP>'] = f"PRIMARY KEY {primary_key_str}"
            else:
                schema['<PRIMARY_KEY_RELATIONSHIP>'] = primary_key_str

            table2schema[table_name] = schema

        cursor.close()
        return table2schema

    def get_table_column_type_mapping(self, lower_case=True):
        mapping = {}
        schema = self.get_schema()
        for tab, val in schema.items():
            for col, col_type in zip(val['columns'], val['columnTypes']):
                if lower_case:
                    col = col.lower()
                    tab = tab.lower()
                if tab not in mapping:
                    mapping[tab] = {}
                mapping[tab][col] = col_type
        return mapping

    def get_lower_case_table_column_to_original_table_column_mapping(self):
        table_mapping = {}
        table_column_mapping = {}
        schema = self.get_schema()
        for tab, val in schema.items():
            clean_tab = tab.strip().lower()
            table_mapping[clean_tab] = tab
            for col, _ in zip(val['columns'], val['columnTypes']):
                if clean_tab not in table_column_mapping:
                    table_column_mapping[clean_tab] = {}
                clean_col = col.strip().lower()
                table_column_mapping[clean_tab][clean_col] = {"table": tab, "column": col}
        return table_column_mapping

    def get_lower_case_to_original_table_mapping(self):
        table_mapping = {}
        schema = self.get_schema()
        for tab, val in schema.items():
            clean_tab = tab.strip().lower()
            table_mapping[clean_tab] = tab
        return table_mapping

    # def get_foreign_key_constraints_after_deleting_column(self, table_name, column_name):
    #     conn = self.in_memory_conn
    #     cursor = conn.cursor()

    #     # Get foreign key constraints for the table
    #     cursor.execute(f"PRAGMA foreign_key_list({table_name})")
    #     foreign_keys = cursor.fetchall()

    #     # Add foreign key constraints back
    #     from_columns_list = []
    #     to_columns_list = []
    #     for fk in foreign_keys:
    #         from_table = fk[2]
    #         from_columns = fk[3].split(",")
    #         to_table = table_name
    #         to_columns = fk[4].split(",")
    #         for to_col, from_col in zip(to_columns, from_columns):
    #             if to_table == table_name and column_name and to_col:
    #                 continue
    #             else:
    #                 from_columns_list.append(from_col)
    #                 to_columns_list.append(to_col)

    #     from_columns_str = ", ".join([f'"{col.strip()}"' for col in from_columns_list])
    #     to_columns_str = ", ".join([f'"{col.strip()}"' for col in to_columns_list])
    #     if from_columns_str.strip() and to_columns_str.strip():
    #         query = f"PRAGMA foreign_key_add({from_table}, {from_columns_str}, {to_table}, {to_columns_str})"
    #     else:
    #         query = ""
    #     cursor.close()
    #     return query

    def get_primary_key_constraints_after_deleting_tab_col(self, table_name, column_name):
        cursor = self.in_memory_conn.cursor()

        # Get the schema of the table
        cursor.execute(f"PRAGMA table_info({table_name})")
        table_info = cursor.fetchall()        
        # Get the column names and their properties
        columns = []
        primary_key_columns_excluding_col_to_delete = []
        for col in table_info:
            col_name = col[1]
            is_primary_key = col[5] == 1
            columns.append(col_name)
            if is_primary_key and column_name is not None and col_name.lower() != column_name.lower():
                primary_key_columns_excluding_col_to_delete.append(col_name)

        # primary keys str
        primary_key_str = ", ".join(primary_key_columns_excluding_col_to_delete)
        cursor.close()
        return primary_key_str
    
    def do_column_contain_duplicate_cell(self, table_name, column_str):
        cursor = self.in_memory_conn.cursor()
        # cursor.execute(f"SELECT COUNT(*) FROM (SELECT {column_str} FROM {table_name} GROUP BY {column_str} HAVING COUNT(*) > 1)")
        if '"' not in column_str:
            column_str = f'"{column_str}"'
        if '"' not in table_name:
            table_name = f'"{table_name}"'
        result = cursor.execute(f"SELECT {column_str}, COUNT(*) AS count FROM {table_name} GROUP BY {column_str} HAVING COUNT(*) > 1;").fetchall()
        if result:
            has_duplicate = True
        else:
            has_duplicate = False
        cursor.close()
        return has_duplicate

    def is_sql_executable(self, sql):
        is_executable = False
        try:
            self.run_sql(sql)
            is_executable = True
        except Exception as ex:
            if isinstance(ex, sqlite3.OperationalError):
                logger.error(f"SQL execution OperationalError: {ex}")
            logger.error(f"SQL execution error: {ex}")
            is_executable = False
        return is_executable

    def delete_column_from_table(self, table_name, column_name):
        # if a column within the current table is deleted, we shall ensure:
        #   1. [done] if the deleted column is primary key of the current table, the primary key constraint is deleted
        #   2. if the deleted column is used as as foreign key in the to table column in any other table, we delete that foreign key constraint
        #   3. if the deleted column is used as a foreign key in the from table column in the current table, we delete the current table's foreign key constraint

        if column_name is None:
            column_name = ""

        cursor = self.in_memory_conn.cursor()
        # logger.debug(f"Schema before modification: {self.get_schema()}")

        lower_case_table_to_original_table_map = self.get_lower_case_to_original_table_mapping()
        original_table_name = lower_case_table_to_original_table_map.get(table_name.lower().strip(), table_name)
        if original_table_name != table_name:
            # logger.warning(f"Table name `{table_name}` does not exist in table. Use original table found: `{original_table_name}`")
            table_name = original_table_name

        # Get the schema of the table
        cursor.execute(f"PRAGMA table_info({table_name})")
        table_info = cursor.fetchall()
        # Get the column names
        columns = [col[1] for col in table_info]
        columns_lower_case = [col.lower() for col in columns]
        # Check if the column exists
        if column_name.lower() not in columns_lower_case:
            logger.error(f"Column '{column_name}' does not exist in table '{table_name}'. The table has columns: {columns}")
            return False

        # if column name lower case exist in the table, find the original column name and set to it
        elif column_name.lower() in columns_lower_case and column_name not in columns:
            for tmp_col_name in columns:
                if column_name.lower() == tmp_col_name.lower():
                    # logger.warning(f"Requested to delete column name: `{column_name}`. It does not exist, column in different casing does exist: `{tmp_col_name}`. Deleting: `{tmp_col_name}`")
                    column_name = tmp_col_name
                    break

        # adjust foreign key constraints for current table
        # TODO: proper handling of column deletion when it is a foreign key or referenced as a foreign key
        # foreign_key_str_to_add_at_the_end = self.get_foreign_key_constraints_after_deleting_column(table_name=table_name, column_name=column_name)

        # Create a temporary table with the desired columns
        temp_table_name = f"temp_{table_name}"
        new_columns = [col for col in columns if col.lower() != column_name.lower()]
        # new_columns_str = ", ".join(new_columns)
        # add single quote around column name to avoid errors like `sqlite3 unrecognized token 18_49_Rating_Share`
        # https://stackoverflow.com/questions/35193495/sqlite-unrecognized-token
        new_columns_str = ", ".join([f'"{col}"' for col in new_columns])

        primary_key_str = self.get_primary_key_constraints_after_deleting_tab_col(table_name, column_name)
        if primary_key_str:
            # if len(new_columns) == 1 and new_columns[0] == primary_key_str:
            if self.do_column_contain_duplicate_cell(table_name=table_name, column_str=primary_key_str):
            # if the new table will only contain one single column and that column will be primary key, we do not want to enforce the primary key constraints
            # otherwise, we will run into error like `UNIQUE constraint failed: temp_singer_in_concert.concert_ID`
            # For example, we create table using `CREATE TABLE temp_singer_in_concert (concert_ID, PRIMARY KEY (concert_ID))'`
            # we insert column values using: `INSERT INTO temp_singer_in_concert ("concert_ID") SELECT "concert_ID" FROM singer_in_concert`
            # concert_ID column contains values: `[(1,), (1,), (1,), (2,), (2,), (3,), (4,), (5,), (5,), (6,)]`
            # we run into errors above: `UNIQUE constraint failed: temp_singer_in_concert.concert_ID`
                create_table_command_str = f"CREATE TABLE {temp_table_name} ({new_columns_str})"
                # logger.warning(f"Primary key contains duplicate values: drop the primary key constraints so that we can delete table. table: {table_name}, primary column: {primary_key_str}")
            else:
                create_table_command_str = f"CREATE TABLE {temp_table_name} ({new_columns_str}, PRIMARY KEY ({primary_key_str}))"
        else:
            create_table_command_str = f"CREATE TABLE {temp_table_name} ({new_columns_str})"
        cursor.execute(create_table_command_str)
        # logger.debug(f"Schema after creating temporary table: {self.get_schema()}")

        # Copy data from the original table to the temporary table
        # columns_str = ", ".join(columns)
        new_columns_str = ", ".join([f'"{col}"' for col in new_columns])
        cursor.execute(f"INSERT INTO {temp_table_name} ({new_columns_str}) SELECT {new_columns_str} FROM {table_name}")

        # Drop the original table
        cursor.execute(f"DROP TABLE {table_name}")
        # logger.debug(f"Schema after dropping original table: {self.get_schema()}")

        # Rename the temporary table to the original table name
        cursor.execute(f"ALTER TABLE {temp_table_name} RENAME TO {table_name}")
        # logger.debug(f"Schema after renaming temporary table: {self.get_schema()}")

        # # if there are still foreign key constraints remaining, add them
        # if foreign_key_str_to_add_at_the_end:
        #     cursor.execute(foreign_key_str_to_add_at_the_end)

        # # adjust the foreign key constraints for the other tables
        # table_names = get_table_names(self.in_memory_conn)
        # for tab_name in table_names:
        #     if tab_name == table_name:
        #         continue
        #     # remove foreign key constraints in other tables by creating new temporary table with foreign constraint removed
        #     self.delete_foreign_key_constraint_from_table(from_table_name=table_name, from_column_name=column_name, table_to_adjust=tab_name)

        self.in_memory_conn.commit()
        cursor.close()
        return True

    def add_column_to_table(self, table_name, column_name, column_type, values):
        add_column_and_insert_values(
            conn=self.in_memory_conn,
            table_name=table_name,
            column_name=column_name,
            column_type=column_type,
            values=values,
        )
    
    def get_grounded_table_column(self, table: str, column: str) -> Dict[str, str]:
        table_lower_case = table.strip().lower()
        column_lower_case = column.strip().lower()
        mapping = self.get_lower_case_table_column_to_original_table_column_mapping()
        # grounded_tab_col = mapping.get(table_lower_case, {}).get(column_lower_case, {"table": table, "column": column})
        grounded_tab_col = mapping.get(table_lower_case, {}).get(column_lower_case, None)
        return grounded_tab_col

    def get_grounded_table(self, table: str) -> str:
        table_lower_case = table.strip().lower()
        mapping = self.get_lower_case_to_original_table_mapping()
        grounded_table = mapping.get(table_lower_case, None)
        return grounded_table

    # def delete_foreign_key_constraint_from_table(self, from_table_name, from_column_name, table_to_adjust):
    #     cursor = self.in_memory_conn.cursor()
    #     # Get foreign key constraints for the table
    #     cursor.execute(f"PRAGMA foreign_key_list({table_to_adjust})")
    #     foreign_keys = cursor.fetchall()

    #     # Add foreign key constraints back
    #     from_columns_list = []
    #     to_columns_list = []
    #     any_foreign_key_to_delete_flag = False
    #     for fk in foreign_keys:
    #         from_table = fk[2]
    #         from_columns = fk[3].split(",")
    #         to_table = table_to_adjust
    #         to_columns = fk[4].split(",")
    #         for to_col, from_col in zip(to_columns, from_columns):
    #             if from_table == from_table_name and from_column_name == from_col:
    #                 any_foreign_key_to_delete_flag = True
    #                 continue
    #             else:
    #                 from_columns_list.append(from_col)
    #                 to_columns_list.append(to_col)
    #     if not any_foreign_key_to_delete_flag:
    #         cursor.close()
    #         return
        
    #     # copy table_to_adjust to a new table with primary key constraint if any
    #     temp_table_name = f"temp_{table_to_adjust}"
    #     cursor.execute(f"PRAGMA table_info({table_to_adjust})")
    #     table_info = cursor.fetchall()
    #     # Get the column names
    #     columns = [col[1] for col in table_info]
    #     columns_str = ", ".join(columns)
    #     primary_key_str = self.get_primary_key_constraints_after_deleting_tab_col(table_name=table_to_adjust, column_name=None)
    #     if primary_key_str:
    #         create_table_command_str = f"CREATE TABLE {temp_table_name} ({columns_str}, PRIMARY KEY ({primary_key_str}))"
    #     else:
    #         create_table_command_str = f"CREATE TABLE {temp_table_name} ({columns_str})"
    #     cursor.execute(create_table_command_str)
    #     # copy data to the new table
    #     cursor.execute(f"INSERT INTO {temp_table_name} ({columns_str}) SELECT {columns_str} FROM {table_to_adjust}")

    #     # formulate the foreign key constraints
    #     from_columns_str = ", ".join([f'"{col.strip()}"' for col in from_columns_list])
    #     to_columns_str = ", ".join([f'"{col.strip()}"' for col in to_columns_list])
    #     if from_columns_str.strip() and to_columns_str.strip():
    #         query = f"PRAGMA foreign_key_add({from_table}, {from_columns_str}, {to_table}, {to_columns_str})"
    #     else:
    #         query = ""

    def delete_cell_from_table(self, tab, col, cell):
        conn = self.in_memory_conn
        try:
            tab_col_mapping = self.get_lower_case_table_column_to_original_table_column_mapping()
            mapping_result = tab_col_mapping[tab.strip().lower()][col.strip().lower()]
            # get the non-lower-cased table name and column name
            original_tab, original_col = mapping_result['table'], mapping_result['column']
            if original_tab != tab or original_col != col:
                # logger.warning(f"Table and column names do not match. Table: {tab}, column: {col}. However, similar name of different casing exist. Used them instead. {original_tab} {original_col}")
                tab, col = original_tab, original_col

            # Create a cursor object
            cursor = conn.cursor()
            # the database cell value sometimes is NOISY.
            # e.g., in flights table, destinationAirport's all cell values start with a white space.
            # the white space does not exist in the SQL and as a result directlying delete it is not possible
            cell_candidate_to_deletes = [cell, f" {cell}", f"{cell} ", f" {cell} ", cell.lower(), f" {cell.lower()}", f"{cell.lower()} "]
            num_rows_deleted = 0
            try:
                # Construct the SQL query to delete the cell value
                # sql = f"DELETE FROM {tab} WHERE {col} = ?" --> this will remove the entire row where cell value equals to this.
                sql = f"UPDATE {tab} SET {col} = NULL WHERE {col} = ?"
                # Execute the SQL query and commit the changes
                for cell2del in cell_candidate_to_deletes:
                    cursor.execute(sql, (cell2del, ))
                    num_rows_deleted += cursor.rowcount
            except Exception as delete_ex:
                # if Error deleting cell value: NOT NULL constraint failed
                # we change the NULL to missing value so that it is not NULL
                if "NOT NULL constraint failed".lower() in str(delete_ex).lower():
                    sql = f"UPDATE {tab} SET {col} = '{self._MISSING_VALUE}' WHERE {col} = ?"
                    # Execute the SQL query and commit the changes
                for cell2del in cell_candidate_to_deletes:
                    cursor.execute(sql, (cell2del, ))
                    num_rows_deleted += cursor.rowcount
                
            cursor.close()
            conn.commit()
            if num_rows_deleted > 0:
                # logger.info(f"Deleted {num_rows_deleted} cell value(s) '{cell}' from column '{col}' in table '{tab}'.")
                return True
            else:
                logger.warning(f"No cell value '{cell}' found in column '{col}' of table '{tab}'. No rows were deleted.")
        except sqlite3.Error as ex:
            logger.error(f"Error deleting cell value: {ex}")
            return False

    def add_cell_to_table_column(self, tab_col_cell_list: List[Dict]):
        # get all cell values from the column
        # find all cell values that are NULL or MISSING_VALUE
        # if there are less than two such cell values, insert MISSING_VALUE to the end
        # set these cell values with the replacement cell values one by one
        # NOTE: assume all cells will be added to the same table and column
        tab, col = tab_col_cell_list[0]['table'], tab_col_cell_list[0]['column']
        grounded_tab_col = self.get_grounded_table_column(table=tab, column=col)
        table_name = grounded_tab_col['table']
        column_name = grounded_tab_col['column']

        all_cell_values = self.get_cell_values(
            only_unique_value=False,
            include_column_type=False,
            ignore_table_column_casing=False,
        )
        original_cells_tab_coll = all_cell_values.get(table_name, {}).get(column_name, None)
        if original_cells_tab_coll is None:
            raise Exception(f"Table and column not found in the database. table: {tab}, column: {col}!")

        all_cell_values_from_tab_col = copy.deepcopy(original_cells_tab_coll)
        null_missing_index_list = []
        for ind, cell in enumerate(all_cell_values_from_tab_col):
            if cell in (None, self._MISSING_VALUE, "NULL"):
                null_missing_index_list.append(ind)
        if len(null_missing_index_list) < len(tab_col_cell_list):
            rows2insert = len(tab_col_cell_list) - len(null_missing_index_list)
            null_missing_index_list.extend([ind + len(all_cell_values_from_tab_col) for ind in range(rows2insert)])
            all_cell_values_from_tab_col.extend([None for _ in range(rows2insert)])
        add_cell_index = 0
        add_cell_expanded = tab_col_cell_list * 100
        for null_missing_ind in null_missing_index_list:
            all_cell_values_from_tab_col[null_missing_ind] = add_cell_expanded[add_cell_index]['value']
            add_cell_index += 1

        values = all_cell_values_from_tab_col

        # insert cell values
        conn = self.in_memory_conn
        cursor = conn.cursor()
        # Insert values into the new column
        update_query = f"UPDATE {table_name} SET {column_name} = ? WHERE rowid = ?"
        insert_query = f"INSERT INTO {table_name} ({column_name}) VALUES (?)"
        for row_id, value in enumerate(values, start=1):
            if row_id < len(original_cells_tab_coll):
                try:
                    cursor.execute(update_query, (value, row_id))
                except sqlite3.Error as e:
                    # logger.error(f"Error updating value. Error: {e}\nvalues to add: {values}\ncurrent value to add: {(value, row_id)}")
                    if row_id < 5:
                        logger.warning(f"Error updating value. Error: {e}\ncurrent value to add: {(value, row_id)}")
            else:
                try:
                    cursor.execute(insert_query, (value,))
                except sqlite3.Error as e:
                    if row_id < 10:
                        logger.error(f"Error inserting value: {e}. Value to insert: {(value, row_id)}")

        cursor.close()
        conn.commit()        


def establish_db_conn(db_id_name, database_main_dir, create_copy = False):
    database_dir = os.path.join(database_main_dir, db_id_name)
    sqlite_file = os.path.join(database_dir, db_id_name+'.sqlite')
    if create_copy:
        new_filename = db_id_name + '_copy.sqlite'
        new_sqlite_file = os.path.join(database_dir, new_filename)
        # copy 
        shutil.copyfile(sqlite_file, new_sqlite_file)
        sqlite_file = new_sqlite_file
    # Connect to the database
    conn = sqlite3.connect(sqlite_file)
    return conn


def extract_col_names(col_info):
    col_names = []
    for info in col_info:
        col_names.append(info[0])
    return col_names


def alter_columns_and_execute_sql(db_info, database_main_dir, new_sql):
    db_name, tab_name, col_name, val = db_info

    # # establish connection
    conn = establish_db_conn(db_name, database_main_dir, create_copy = True)
    # # remove the column from 
    # result = conn.execute('SELECT TYPE_NAME({:s}) from {:s}'.format(col_name, tab_name)).fetchall()
    # print(result)

    result = conn.execute('PRAGMA table_info([{:s}]);'.format(tab_name))
    result = conn.execute('select * from {:s} where {:s} = {:s}'.format(tab_name, col_name, val)).fetchall()
    # print('Before nulling the columns:', result)

    # rename old column
    conn.execute('ALTER TABLE {:s} rename {:s} to {:s}_old'.format(tab_name, col_name, col_name))
    # TODO: Add new columns
    conn.execute('ALTER TABLE {:s} ADD {:s} NULL'.format(tab_name, col_name))
    # copy information into new column
    conn.execute('UPDATE {:s} SET {:s} = {:s}'.format(tab_name, col_name, "{:s}_old".format(col_name)))
    # drop the old column
    conn.execute('ALTER TABLE {:s} DROP COLUMN {:s}'.format(tab_name, "{:s}_old".format(col_name)))


    # set new column to be null
    conn.execute("update {:s} set {:s}=null where {:s}={:s}".format(tab_name, col_name, col_name, val))
    # double check the table
    result = conn.execute('select * from {:s} where {:s} = {:s}'.format(tab_name, col_name, val)).fetchall()
    if len(result) != 0:
        print('Error - Length not zero')
    
    # Execute SQL
    try:
        result = conn.execute(new_sql)
        col_names = extract_col_names(result.description)
        full_results = result.fetchall()
    except Exception as e:
        print(e)
        print(new_sql)
        col_names, full_results = None, None
    return col_names, full_results


if __name__ == "__main__":
    main_db_dir = ".vscode/combined_data_all/spider/database"
    # for db_name in os.listdir(main_db_dir):
    #     # db_path = "academic"
    #     # db_path = "decoration_competition"
    #     db = DbWithModification(db_name, main_db_dir, None)
    #     tab_col_cells = db.get_cell_values(only_unique_value=False)
    #     print(tab_col_cells)
    #     db.close()

    # test column deletion
    db_id = "concert_singer"
    online_db = DbWithModification(db_id, main_db_dir, None)

    db_schema = online_db.get_schema()
    online_db.get_cell_values()
    # test deleting foreign key
    online_db.delete_column_from_table(table_name="concert", column_name="Stadium_ID")
    assert "Stadium_ID" not in online_db.get_schema()['concert']
    # test deleting primary key
    online_db.delete_column_from_table(table_name="stadium", column_name="Stadium_ID")
    assert "Stadium_ID" not in online_db.get_schema()['stadium']
    print(online_db.get_schema())
