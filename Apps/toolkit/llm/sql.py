import os
import json
import datetime
import pytz
import time
import pandas as pd
import logging
from typing import Literal
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool


class MySQLDatabase:
	def __init__(
		self,
		dbname: str = os.getenv("SQL_DB"),
		user: str = os.getenv("SQL_USER"),
		password: str = os.getenv("SQL_PASSWORD"),
		host: str = "postgres" if os.getenv("IN_PROD") else os.getenv("SQL_HOST"),
		port: str = os.getenv("SQL_PORT"),
		min_conn: int = 1,
		max_conn: int = 1,
	):
		self.dbname = dbname
		self.user = user
		self.password = password
		self.host = host
		self.port = port
		self.logger = logging.getLogger(__name__)

		self.pool = SimpleConnectionPool(
			minconn=min_conn,
			maxconn=max_conn,
			dsn="dbname={dbname} user={user} password={password} host={host} port={port}".format(
				dbname=dbname, user=user, password=password, host=host, port=port
			)
		)

	def execute_query(
			self,
			query: str,
			params: tuple = None,
			fetch_option: Literal["one", "many", "all"] = "all",
		):
			conn: psycopg2.extensions.connection = None
			cur: RealDictCursor = None
			try:
					conn = self.pool.getconn()
					cur = conn.cursor(cursor_factory=RealDictCursor)
					if params:
							cur.execute(query, params)
					else:
							cur.execute(query)
					if fetch_option == "one":
							return cur.fetchone()
					elif fetch_option == "many":
							return cur.fetchmany()
					elif fetch_option == "all":
							return cur.fetchall()
			except (Exception, psycopg2.DatabaseError) as error:
					self.logger.error(f"Error: {error}")
					return None
			finally:
					if cur is not None:
							cur.close()
					if conn is not None:
							self.pool.putconn(conn)

	def execute_commit(self, query, params=None):
		conn: psycopg2.extensions.connection = None
		cur: RealDictCursor = None
		try:
			conn = self.pool.getconn()
			cur = conn.cursor()
			if params:
				cur.execute(query, params)
			else:
				cur.execute(query)
			conn.commit()
			return cur.rowcount
		except (Exception, psycopg2.DatabaseError) as error:
			self.logger.error(f"Error: {error}")
			if conn is not None:
				conn.rollback()
			return None
		finally:
			if cur is not None:
				cur.close()
			if conn is not None:
				self.pool.putconn(conn)
	
	def get_uri(self):
				return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"



class MySQLTable:
		def __init__(self, name: str, schema: list[str], db: MySQLDatabase):
				self.name = name
				self.schema = schema
				self.db = db
				"""
				schema_users = [
					"id SERIAL",
					"username VARCHAR(50) UNIQUE NOT NULL",
					"email VARCHAR(255) UNIQUE NOT NULL",
					"password_hash VARCHAR(128) NOT NULL",
					"created_at TIMESTAMP DEFAULT NOW()",
					"updated_at TIMESTAMP DEFAULT NOW()",
					"PRIMARY KEY (id)"
				]
				"""

				# Extract column names from schema and store them in a separate list
				self.col_names = [col.split()[0] for col in schema if col.split()[0] not in ["PRIMARY", "FOREIGN"]]

				# Store the timezone of the local system
				self.tz = datetime.timezone.utc if datetime.timezone.utc.tzname(None) == "UTC" else datetime.timezone(datetime.timedelta(seconds=time.timezonezone(None)))

		def get_schema(self, is_get_all=False):
				"""Get the schema of the table.

				Args:
						is_get_all (bool, optional): If True, returns all column information. If False, returns only essential column information. Defaults to False.

				Returns:
						list: A list of dictionaries containing column information.
				"""
				query = f"""
						SELECT *
						FROM information_schema.columns
						WHERE table_name = '{self.name}'
						ORDER BY ordinal_position;
				"""
				schema = self.db.execute_query(query, fetch_option="all")
				rows = []
				for row in schema:
						if is_get_all:
								row_info = {key: val for key, val in row.items() if val is not None}
						else:
								row_info = {
										'table_catalog': row['table_catalog'],
										'table_schema': row['table_schema'],
										'table_name': row['table_name'],
										'column_name': row['column_name'],
										'is_nullable': row['is_nullable'],
										'data_type': row['data_type']
								}
						rows.append(row_info)
				return rows
	
		def create(self):
				query = f"CREATE TABLE IF NOT EXISTS {self.name} ("
				for i, col in enumerate(self.schema):
						query += f"{col} "
						if i < len(self.schema) - 1:
								query += ", "
				query += ")"
				self.db.execute_commit(query)

		def insert(self, data: dict):
				# Exclude the 'id SERIAL', 'created_at', and 'updated_at' columns from the list of columns and placeholders
				cols = ", ".join(col for col in self.col_names if col not in ["id", "created_at", "updated_at"])
				placeholders = ", ".join("%s" for col in self.col_names if col not in ["id", "created_at", "updated_at"])

				# Convert the timestamps to JSON objects with a timezone offset
				now = datetime.datetime.now(self.tz)
				data["created_at"] = self._to_json_date(now)
				data["updated_at"] = self._to_json_date(now)

				query = f"INSERT INTO {self.name} ({cols}) VALUES ({placeholders})"
				self.db.execute_commit(query, tuple(data[col] for col in self.col_names if col not in ["id", "created_at", "updated_at"]))

		def delete_by_col(self, col: str, value: str):
				"""Delete rows from the table that match a given column value."""
				condition = f"{col} = %s"
				params = (value,)
				self.delete(condition, params)

		def update_by_col(self, col: str, value: str, data: dict):
				"""Update rows in the table that match a given column value."""
				condition = f"{col} = %s"
				params = (value,)

				# Generate a list of column-value pairs with placeholders
				col_value_pairs = [f"{col}=%s" for col in data]

				# Substitute the placeholders with the actual values
				for i, col_value_pair in enumerate(col_value_pairs):
						col_value_pairs[i] = col_value_pair % (data[col],)

				# Join the column-value pairs into a single string
				cols = ", ".join(col_value_pairs)

				query = f"UPDATE {self.name} SET {cols} WHERE {condition}"
				self.db.execute_commit(query, params + tuple(data[col] for col in data))

		def delete(self, condition: str, params: tuple = None):
				query = f"DELETE FROM {self.name} WHERE {condition}"
				self.db.execute_commit(query, params)

		def update(self, data: dict, condition: str, params: tuple = None):
				cols = ", ".join(f"{col}=%s" for col in data)
				query = f"UPDATE {self.name} SET {cols} WHERE {condition}"
				all_params = tuple(data[col] for col in data) + (params if params is not None else ())
				self.db.execute_commit(query, all_params)

		def query(self, query: str, params: tuple = None, fetch_option: Literal["one", "many", "all"] = "all"):
				cur = self.db.execute_query(query, params)
				if fetch_option == "one":
						return cur.fetchone()
				elif fetch_option == "many":
						return cur.fetchmany()
				elif fetch_option == "all":
						return cur.fetchall()

		def _to_json_date(self, dt: datetime.datetime):
				"""Convert a datetime.datetime object to a JSON object with a timezone offset."""
				return json.dumps({"$date": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + self.tz.tzname(None)})

		def _from_json_date(self, json_str: str):
				"""Convert a JSON object with a timezone offset to a datetime.datetime object."""
				obj = json.loads(json_str)
				dt_str = obj["$date"][:-6] + obj["$date"][-5:]
				dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
				tz_str = obj["$date"][-6:]
				tz = pytz.timezone(tz_str) if tz_str != "UTC" else datetime.timezone.utc
				return dt.replace(tzinfo=tz)

		def insert_from_dataframe(self, df: pd.DataFrame):
				"""Insert rows from a Pandas DataFrame into the table."""
				# Exclude the 'id SERIAL', 'created_at', and 'updated_at' columns from the DataFrame
				df = df.loc[:, df.columns.intersection(set(self.col_names) - {"id", "created_at", "updated_at"})]

				# # Check if all the required columns are present in the DataFrame
				# required_cols = {"username", "email", "password_hash"}
				# if not required_cols.issubset(df.columns):
				# 		raise ValueError("Missing required columns in the DataFrame")

				# Convert the timestamps to PostgreSQL timestamp strings
				now = datetime.datetime.now(self.tz)
				if "created_at" in self.col_names:
					df["created_at"] = [self._to_pg_timestamp(now)] * len(df)
				if "updated_at" in self.col_names:
					df["updated_at"] = [self._to_pg_timestamp(now)] * len(df)

				# Generate a list of column names
				cols = ", ".join(df.columns)

				# Construct the SQL query with the %s placeholders
				query = f"INSERT INTO {self.name} ({cols}) VALUES ({', '.join(['%s'] * len(df.columns))})"

				# Convert the DataFrame to a list of tuples
				values = [tuple(x) for x in df.to_numpy()]

				# Execute the query with the list of tuples
				for value in values:
						self.db.execute_commit(query, value)

		def _to_pg_timestamp(self, dt: datetime.datetime):
				"""Convert a datetime.datetime object to a PostgreSQL timestamp string."""
				return dt.astimezone(self.tz).strftime('%Y-%m-%d %H:%M:%S%z')

		def _from_pg_timestamp(self, timestamp_str: str):
				"""Convert a PostgreSQL timestamp string to a datetime.datetime object."""
				dt = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S%z')
				return dt.replace(tzinfo=self.tz)
			
		def insert_from_csv(self, csv_path: str):
				"""Insert rows from a CSV file into the table."""

				# Read the CSV file into a Pandas DataFrame
				df = pd.read_csv(csv_path)

				# Insert the rows from the DataFrame into the table
				self.insert_from_dataframe(df)

		def insert_from_excel(self, excel_path: str):
				"""Insert rows from a Excel file into the table."""

				# Read the CSV file into a Pandas DataFrame
				df = pd.read_excel(excel_path)

				# Insert the rows from the DataFrame into the table
				self.insert_from_dataframe(df)

		def get_discrete_values_col(self, col: str):
				"""Get the distinct values of a specified column."""
				query = f"SELECT DISTINCT {col} FROM {self.name}"
				results = self.db.execute_query(query, fetch_option="all")
				return [row[col] for row in results]

		def get_discrete_values_cols(self, cols: list[str]):
				"""Get the distinct values of a specified list of columns."""
				cols_str = ", ".join(cols)
				query = f"SELECT DISTINCT {cols_str} FROM {self.name}"
				results = self.db.execute_query(query, fetch_option="all")
				return [{col: row[col] for col in cols} for row in results]

...