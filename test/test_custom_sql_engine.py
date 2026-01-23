import os
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from custom_sql_engine import add_column_and_insert_values, DbWithModification
from utils import (
    get_lexically_similar_columns_from_schema
)


def create_dummy_database(temp_dir):
    database_main_dir = temp_dir.name
    db_id_name = "dummy_db"

    # Create a new SQLite database
    db_fp = os.path.join(database_main_dir, db_id_name, db_id_name + ".sqlite")
    os.makedirs(os.path.dirname(db_fp), exist_ok=True)
    conn = sqlite3.connect(db_fp)
    
    cursor = conn.cursor()

    # Create the first table
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            age INTEGER,
            city TEXT,
            country TEXT
        )
    """)

    # Insert dummy data into the users table
    cursor.executemany("""
        INSERT INTO users (name, email, age, city, country)
        VALUES (?, ?, ?, ?, ?)
    """, [
        ('John Doe', 'john.doe@example.com', 35, 'New York', 'USA'),
        ('Jane Smith', 'jane.smith@example.com', 28, 'London', 'UK'),
        ('Michael Johnson', 'michael.johnson@example.com', 42, 'Sydney', 'Australia'),
        ('Emily Davis', 'emily.davis@example.com', 31, 'Toronto', 'Canada'),
        ('David Wilson', 'david.wilson@example.com', 27, 'Berlin', 'Germany')
    ])

    # Create the second table
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT,
            price REAL,
            category TEXT,
            stock INTEGER
        )
    """)

    # Insert dummy data into the products table
    cursor.executemany("""
        INSERT INTO products (name, description, price, category, stock)
        VALUES (?, ?, ?, ?, ?)
    """, [
        ('T-Shirt', 'Cotton t-shirt', 19.99, 'Clothing', 100),
        ('Jeans', 'Denim jeans', 49.99, 'Clothing', 75),
        ('Laptop', 'High-performance laptop', 999.99, 'Electronics', 25),
        ('Headphones', 'Wireless headphones', 79.99, 'Electronics', 50),
        ('Book', 'Best-selling novel', 14.99, 'Books', 200)
    ])

    # Create the third table
    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            total_price REAL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # Insert dummy data into the orders table
    today = date.today()
    cursor.executemany("""
        INSERT INTO orders (user_id, product_id, quantity, order_date, total_price)
        VALUES (?, ?, ?, ?, ?)
    """, [
        (1, 1, 2, today.strftime('%Y-%m-%d'), 39.98),
        (2, 3, 1, (today - timedelta(days=3)).strftime('%Y-%m-%d'), 999.99),
        (3, 2, 1, (today - timedelta(days=7)).strftime('%Y-%m-%d'), 49.99),
        (4, 4, 2, (today - timedelta(days=10)).strftime('%Y-%m-%d'), 159.98),
        (5, 5, 3, (today - timedelta(days=15)).strftime('%Y-%m-%d'), 44.97)
    ])

    conn.commit()
    conn.close()

    return database_main_dir, db_id_name


class TestDbWithModification(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dummy_database_main_dir, self.dummy_db_id_name = create_dummy_database(self.temp_dir)
        
        self.schema_modification = {"removeColumn": [{"table": "users", "column": "id"}, {"table": "orders", "column": "user_id"}]}

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_delete_column_from_table(self):
        # check if the column existing without schema modification
        test_db = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, {})
        for tab_col in self.schema_modification["removeColumn"]:
            self.assertIn(tab_col["column"], test_db.get_cell_values(only_unique_value=True)[tab_col['table']])
        test_db = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, self.schema_modification)
        for tab_col in self.schema_modification["removeColumn"]:
            self.assertNotIn(tab_col["column"], test_db.get_cell_values(only_unique_value=True)[tab_col['table']])

    def test_add_column_and_insert_values(self):
        # Add a new column and insert values
        test_db = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, self.schema_modification)
        test_values = ["value1", "value2", "value3"]
        existing_value, existing_column, existing_table = "Best-selling novel", "description", "products"
        # check the cell values do not exist before adding
        self.assertIn(existing_value, test_db.get_cell_values(only_unique_value=True)[existing_table][existing_column])

        # check new column does not exist before insertion
        self.assertNotIn("new_column", test_db.get_cell_values(only_unique_value=True)["orders"])
        # add new column and insert values
        test_db.add_column_to_table("orders", "new_column", "TEXT", values=test_values)
        # check the cell values exist after adding
        self.assertListEqual(test_values, test_db.get_cell_values(only_unique_value=False)["orders"]["new_column"][:3])

    def test_is_sql_executable(self):
        test_db = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, self.schema_modification)
        self.assertTrue(test_db.is_sql_executable("SELECT * FROM users"))
        self.assertFalse(test_db.is_sql_executable("SELECT * FROM users WHERE id = 1"))
        self.assertFalse(test_db.is_sql_executable("Hello world;"))

    def test_get_cell_values(self):
        test_db = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, self.schema_modification)
        cell_with_column_type = test_db.get_cell_values(only_unique_value=True, include_column_type=True)
        self.assertIn("products", cell_with_column_type)
        self.assertIn('Cotton t-shirt', cell_with_column_type['products']['description']['cell_values'])
        self.assertIn('cell_values', cell_with_column_type['products']['description'])
        self.assertIn('type', cell_with_column_type['products']['description'])

        cell_without_column_type = test_db.get_cell_values(only_unique_value=True, include_column_type=False)
        self.assertIn("products", cell_without_column_type)
        self.assertIn('Cotton t-shirt', cell_without_column_type['products']['description'])
        self.assertNotIn('cell_values', cell_without_column_type['products']['description'])
        self.assertNotIn('type', cell_without_column_type['products']['description'])

    def test_delete_cell_from_table(self):
        cell_mod = {
            "removeCell": [{
                "table": "users",
                "column": "name",
                "value": "John Doe"
            }]
        }
        test_db = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, schema_modification={})
        # check if the cell existing before schema modification
        self.assertIn("John Doe", test_db.get_cell_values(only_unique_value=True)["users"]["name"])
        self.assertIn("Emily Davis", test_db.get_cell_values(only_unique_value=True)["users"]["name"])
        test_db_with_cell_removed = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, cell_mod)
        # check if the cell existing before schema modification
        remaining_cells = test_db_with_cell_removed.get_cell_values(only_unique_value=True)
        self.assertNotIn("John Doe", remaining_cells["users"]["name"])
        self.assertIn("Emily Davis", remaining_cells["users"]["name"])
        # ensure we only delete the cell value from specified column and table
        # cell values in the other column of the table shall not be affected
        self.assertIn("john.doe@example.com", remaining_cells['users']['email'])

        # test deleting non-existing cell values
        nonexist_cell_mod = {
            "removeCell": [{
                "table": "users",
                "column": "name",
                "value": "John"
            }]
        }
        test_db_with_cell_removed = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, nonexist_cell_mod)
        # check if the cell existing before schema modification
        remaining_cells = test_db_with_cell_removed.get_cell_values(only_unique_value=True)
        self.assertIn("John Doe", remaining_cells["users"]["name"])
        self.assertIn("Emily Davis", remaining_cells["users"]["name"])

        # test lower case removal
        casing_cell_mod = {
            "removeCell": [{
                "table": "USERS",
                "column": "Name",
                "value": "John Doe"
            }]
        }
        test_db_with_cell_removed = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, casing_cell_mod)
        # check if the cell existing before schema modification
        remaining_cells = test_db_with_cell_removed.get_cell_values(only_unique_value=True)
        self.assertNotIn("John Doe", remaining_cells["users"]["name"])
        self.assertIn("Emily Davis", remaining_cells["users"]["name"])

    def test_add_cell_value_to_table_column(self):
        cell_remove = [{
            "table": "users",
            "column": "name",
            "value": "John Doe"
        }]
        cell_add = [
            {"table": "users", "column": "name", "value": "ambiguous John 1"},
            {"table": "users", "column": "name", "value": "ambiguous John 2"}
        ]
        schema_mod = {
            "removeCell": cell_remove,
            "addCell": cell_add
        }
        test_db_with_cell_replacement = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, schema_modification=schema_mod)
        modified_cells = test_db_with_cell_replacement.get_cell_values(only_unique_value=False)
        print(modified_cells)
        self.assertIn("ambiguous John 1", modified_cells['users']['name'])
        self.assertIn("ambiguous John 2", modified_cells['users']['name'])
        self.assertNotIn("John Doe", modified_cells['users']['name'])

        original_db = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, schema_modification={})
        original_cells = original_db.get_cell_values(only_unique_value=False)
        self.assertIn("John Doe", original_cells['users']['name'])
        self.assertNotIn("ambiguous John 1", original_cells['users']['name'])
        self.assertNotIn("ambiguous John 2", original_cells['users']['name'])
        print(original_cells)

    def test_get_oracle_cell_values_for_amb_values_within_column(self):
        cell_remove = [{
            "table": "users",
            "column": "name",
            "value": "John Doe"
        }]
        cell_add = [
            {"table": "users", "column": "name", "value": "ambiguous John 1"},
            {"table": "users", "column": "name", "value": "ambiguous John 2"}
        ]
        schema_mod = {
            "removeCell": cell_remove,
            "addCell": cell_add
        }
        test_db_with_cell_replacement = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, schema_modification=schema_mod)
        cell_values = test_db_with_cell_replacement.get_oracle_cell_values_for_amb_values_within_column()
        cell_values_of_interest = cell_values[cell_remove[0]['table']][cell_remove[0]['column']][:2]
        self.assertIn(schema_mod['addCell'][0]['value'], cell_values_of_interest)
        self.assertIn(schema_mod['addCell'][1]['value'], cell_values_of_interest)

    def test_get_oracle_cell_values_for_amb_values_across_column(self):
        col_remove = [{
            "table": "users",
            "column": "name",
            "value": "John Doe"
        }]
        col_add = [
            {"table": "users", "column": "name_111", "value": ["a", "b", "c", "John Doe"], "type": "TEXT"},
            {"table": "users", "column": "name_222", "value": ["c", "f", "e", "John Doe"], "type": "TEXT"},
            # {"table": "users", "column": "name_111", "value": ["ambiguous John 1"], "type": "TEXT"},
            # {"table": "users", "column": "name_222", "value": ["ambiguous John 2"], "type": "TEXT"},
        ]
        schema_mod = {
            "removeColumn": col_remove,
            "addColumn": col_add
        }
        test_db_with_cell_replacement = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, schema_modification=schema_mod)
        cell_values = test_db_with_cell_replacement.get_oracle_cell_values_for_amb_values_across_column()

        self.assertIn(col_remove[0]['value'], cell_values[col_add[0]['table']][col_add[0]['column']])
        self.assertIn(col_remove[0]['value'], cell_values[col_add[1]['table']][col_add[1]['column']])
        self.assertNotIn(col_remove[0]['column'], cell_values[col_remove[0]['table']])

        cell_values_non_oracle = test_db_with_cell_replacement.get_cell_values()
        self.assertIn(col_remove[0]['value'], cell_values_non_oracle[col_add[0]['table']][col_add[0]['column']])
        self.assertNotIn(col_remove[0]['value'], cell_values_non_oracle[col_add[0]['table']][col_add[0]['column']][:2])

    def test_add_columns_to_table(self):
        pass

    def test_get_cell_values_sorted(self):
        test_db_with_cell_replacement = DbWithModification(self.dummy_db_id_name, self.dummy_database_main_dir, schema_modification={})
        cell_values = test_db_with_cell_replacement.get_cell_values(sort_order="alphabet")
        prev_table_key = "a"
        for table_key in cell_values.keys():
            self.assertTrue(table_key > prev_table_key)
            prev_table_key = table_key
            column_cell_dict = cell_values[table_key]
            prev_col_key = "a"
            for col_key in column_cell_dict.keys():
                self.assertTrue(col_key > prev_col_key)
                prev_col_key = col_key

def test_get_lexically_similar_columns_from_schema():
    tab_col_cells = {
        "users": {
            "id": ["1", "2", "3", "4", "5"],
            "name": ["John", "Jane", "Michael", "Emily", "David"],
            "email": ["john.doe@example.com", "jane.smith@example.com", "michael.johnson@example.com", "emily.davis@example.com", "david.wilson@example.com"],
            "age": ["35", "28", "42", "31", "27"],
            "city": ["New York", "London", "Sydney", "Toronto", "Berlin"],
            "country": ["USA", "UK", "Australia", "Canada", "Germany"]
        },
        "orders": {
            "id": ["1", "2", "3", "4", "5"],
            "user_id": ["1", "2", "3", "4", "5"],
            "product_id": ["1", "2", "3", "4", "5"],
            "quantity": ["2", "1", "1", "2", "3"],
        },
        "customers": {
            "id": ["1", "2", "3", "4", "5"],
            "name": ["John", "Jane", "Michael", "Emily", "David"],
            "emails": ["john.doe@example.com", "jane.smith@example.com", "michael.johnson@example.com", "emily.davis@example.com", "david.wilson@example.com"],
            "age": ["35", "28", "42", "31", "27"],
            "city": ["New York", "London", "Sydney", "Toronto", "Berlin"],
            "country": ["USA", "UK", "Australia", "Canada", "Germany"]
        }
    }
    tab_col_list = [
        {"table": "users", "column": "id"},
        {"table": "customers", "column": "name"},
        {"table": "users", "column": "email"}
    ]
    expected_similar_columns = [
        [],
        [],
        [{"table": "customers", "column": "emails"}],
    ]
    for x, y in zip(tab_col_list, expected_similar_columns):
        pred_y = get_lexically_similar_columns_from_schema(x, tab_col_cells=tab_col_cells)
        assert all(str(a) == str(b) for a, b in zip(pred_y, y))


if __name__ == "__main__":
    # unittest.main()
    test_get_lexically_similar_columns_from_schema()
    test = TestDbWithModification()
    test.setUp()
    test.test_get_cell_values_sorted()
    test.test_get_oracle_cell_values_for_amb_values_within_column()
    test.test_get_oracle_cell_values_for_amb_values_across_column()
    test.test_add_cell_value_to_table_column()
    test.test_delete_cell_from_table()
    test.test_is_sql_executable()
    test.test_add_column_and_insert_values()
    test.test_delete_column_from_table()
    test.test_get_cell_values()
