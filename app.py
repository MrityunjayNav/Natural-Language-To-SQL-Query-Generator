from flask import Flask,request, jsonify, render_template
from langchain_ollama import OllamaLLM
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from ydata_profiling import ProfileReport
import psycopg2
from dotenv import load_dotenv
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime

load_dotenv()  # This must be BEFORE os.getenv()

# --------------------- LOGGING CONFIGURATION --------------------

def setup_logging():
    """Configure comprehensive logging for the application"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler with rotation
            RotatingFileHandler(
                'logs/app.log',
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Create specific loggers for different components
    db_logger = logging.getLogger('database')
    llm_logger = logging.getLogger('llm')
    api_logger = logging.getLogger('api')
    
    # Set log levels
    db_logger.setLevel(logging.INFO)
    llm_logger.setLevel(logging.INFO)
    api_logger.setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging()
db_logger = logging.getLogger('database')
llm_logger = logging.getLogger('llm')
api_logger = logging.getLogger('api')

logger.info("Application starting up...")

# Log environment variables (without sensitive data)
logger.info(f"DB_HOST: {os.getenv('DB_HOST')}")
logger.info(f"DB_PORT: {os.getenv('DB_PORT')}")
logger.info(f"DB_USER: {os.getenv('DB_USER')}")
logger.info(f"DB_NAME: {os.getenv('DB_NAME')}")


print("DB_HOST:", os.getenv("DB_HOST"))
print("DB_PORT:", os.getenv("DB_PORT"))
print("DB_USER:", os.getenv("DB_USER"))
print("DB_PASSWORD:", os.getenv("DB_PASSWORD"))
print("DB_NAME:", os.getenv("DB_NAME"))


app = Flask(__name__)
CORS(app)

# --------------------- COLUMN DISPLAY MAPPING ---------------------

COLUMN_MAPPINGS = {
    'id': 'ID',
    'name': 'Name',
    'email': 'Email',
    'message': 'Message',
    'createdAt': 'Created At',
    'documentId': 'Document ID',
    'reminderDate': 'Reminder Date',
    'sent': 'Sent',
    'updatedAt': 'Updated At',
    'checksum': 'Checksum',
    'finished_at': 'Finished At',
    'migration_name': 'Migration Name',
    'logs': 'Logs',
    'rolled_back_at': 'Rolled Back At',
    'started_at': 'Started At',
    'applied_steps_count': 'Applied Steps Count',
    'filename': 'File Name',
    'filepath': 'File Path',
    'mimetype': 'Mime Type',
    'size': 'Size',
    'url': 'URL',
    'uploaded_at': 'Uploaded At',
    'folder': 'Folder',
    'description': 'Description',
    'owned_by': 'Owned By',
    'maintained_by': 'Maintained By',
    'start_date':'Start Date',
    'expiration_date':'Expiration Date',
    'uploaded_date':'Uploaded Date',
    'updated_at':'Updated At',
    'archived':'Archieved',
    'maintained_by_email':'Maintained By Email',
    'owned_by_email':'Owned By Email',
    'document_id':'Revised Document ID',
    'action_type':'Action Type',
    'action_timestamp':'Action Time',
    'user_id':'User ID',
    'user_name':'User Name',
    'old_data':'Old Data',
    'new_data':'New Data',
    'comments':'Comments'
}

class HeaderMapper:
    def __init__(self, mappings=None):
        self.mappings = mappings or COLUMN_MAPPINGS
        logger.info("HeaderMapper initialized with mappings")

    def get_display_name(self, column_name):
        display_name = self.mappings.get(column_name, self._format_column_name(column_name))
        logger.debug(f"Column mapping: {column_name} -> {display_name}")
        return display_name

    def _format_column_name(self, column_name):
        formatted = ' '.join(word.capitalize() for word in column_name.split('_'))
        logger.debug(f"Formatted column name: {column_name} -> {formatted}")
        return formatted

    def transform_data_result(self, data_list):
        if not data_list:
            logger.info("No data to transform")
            return []
        
        logger.info(f"Transforming {len(data_list)} rows of data")
        transformed_data = []
        for row in data_list:
            transformed_row = {
                self.get_display_name(k): v for k, v in row.items()
            }
            transformed_data.append(transformed_row)
        
        logger.info(f"Successfully transformed {len(transformed_data)} rows")
        return transformed_data

# Initialize the header mapper and LLMs
header_mapper = HeaderMapper()

# Initialize LLMs once to avoid recreating on every request
try:
    llm_logger.info("Initializing Ollama LLM...")
    sql_llm = OllamaLLM(
    model="mistral:7b-instruct",
    base_url="http://127.0.0.1:11434"
)

    llm_logger.info("Ollama LLM initialized successfully")
except Exception as e:
    llm_logger.error(f"Error initializing LLMs: {e}")
    print(f"Error initializing LLMs: {e}")
    sql_llm = None

# --------------------- LLM CONTEXT & HELPER ---------------------

# The final, corrected version of the context variable with all fixes
context = """
You are an expert PostgreSQL query generator. Your task is to convert a user's question into a single, valid PostgreSQL query.

---
## Context and Rules

1.  **Database Schema:** The database name is 'documents_management'. Here are the tables and their columns.
    * **"Feedback"**: id (integer), name (text), email (text), message (text), "createdAt" (timestamp)
    * **"Reminder"**: id (integer), "documentId" (integer), "reminderDate" (timestamp), sent (boolean), "createdAt" (timestamp), "updatedAt" (timestamp)
    * **"_prisma_migrations"**: id (string), checksum (string), finished_at (timestamp), migration_name (string), logs (text), rolled_back_at (timestamp), started_at (timestamp), applied_steps_count (integer)
    * **"file_uploads"**: id (integer), filename (text), filepath (text), mimetype (text), size (integer), url (text), "uploaded_at" (timestamp), **folder (text, The category or 'Type' of the document, e.g., 'Contracts')**, description (text), owned_by (text, The full name of the owner), maintained_by (text, The full name of the maintainer), start_date (date), expiration_date (date), uploaded_date (date), updated_at (timestamp), archived (boolean), maintained_by_email (text, The email of the maintainer), owned_by_email (text, The email of the owner)
    * **"revision_history"**: id (integer), document_id (integer), action_type (text), action_timestamp (timestamp), user_id (text), user_name (text), old_data (jsonb), new_data (jsonb), comments (text)

2.  **Foreign Key Relationships (for JOINs):**
    * `"Reminder"."documentId"` refers to `"file_uploads"."id"`
    * `"revision_history"."document_id"` refers to `"file_uploads"."id"`

---
## Strict Query Generation Rules

1.  **Handle Greetings & Irrelevant Input (Highest Priority):** If the user's input is a simple greeting (like "hello", "hi", "how are you", "thanks"), a question about you ("who are you"), or is completely irrelevant gibberish ("jfsdkhf", "sdh"), do not generate a query or an error. Your ONLY response must be the single word: **`GREETING`**.

2.  **Adhere to Schema:** You **MUST** use the exact table and column names provided in the schema above. Do not query any tables that are not explicitly listed in the 'Database Schema' section (for example, do not invent a 'users' table). If a query requires a value that isn't provided (like an email for a name), return an ERROR asking for the missing information.

3.  **MANDATORY Quoting:** You **MUST** use double quotes (`"`) around all table names and any column names that are camelCase or mixed-case.

4.  **Dynamic Dates:** For any queries involving the current time or relative dates (like 'today', 'yesterday', 'next month'), you **MUST** use PostgreSQL's date functions. Use `NOW()` for the current timestamp, `CURRENT_DATE` for today's date, and `INTERVAL` for date arithmetic.

5.  **Text Search:** For case-insensitive text searching, **ALWAYS** use the `ILIKE` operator.

6.  **Joins:** When a query requires joining tables, use the relationships defined in the "Foreign Key Relationships" section to ensure the `JOIN` condition is correct.

7.  **Generate Errors for Ambiguous Queries:** This rule applies **ONLY if the input is not a greeting**. If a user's request looks like a real query but is ambiguous or asks for a column that does not exist, return a single line of text starting with `ERROR:` that explains the problem. If an UPDATE request is ambiguous about which column to change (e.g., "change the file to 'Agreements'"), return an ERROR asking the user to specify the column, like 'ERROR: Please specify which field to change (e.g., title, type, description)'.

8.  **Safety:** Unless the user's request is explicitly about changing data (update, delete), you **MUST** generate a `SELECT` statement.

---
## Query Examples

User Question: "hello"
Your SQL Response: GREETING

User Question: "tell me about the company employees"
Your SQL Response: ERROR: The table 'employees' does not exist in the schema.

User Question: "Show me all files that have been archived"
Your SQL Response: SELECT * FROM "file_uploads" WHERE "archived" = true;

User Question: "Count the number of documents owned by each person"
Your SQL Response: SELECT owned_by_email, COUNT(id) FROM "file_uploads" GROUP BY owned_by_email;

User Question: "Which documents are expiring next month?"
Your SQL Response: SELECT filename, expiration_date FROM "file_uploads" WHERE expiration_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '1 month';

User Question: "Update the expiration date for document ID 25 to '2027-01-01'"
Your SQL Response: UPDATE "file_uploads" SET "expiration_date" = '2027-01-01' WHERE "id" = 25;

---
## Output Format

**CRITICAL:** Your entire response must be **ONLY the raw SQL query** (or the special words `GREETING` or `ERROR:` as defined in the rules). Do not include any explanations, markdown, or other text.
"""

def get_assistant_response(question, context):
    llm_logger.info(f"Processing question: {question}")
    
    if not sql_llm:
        llm_logger.error("SQL LLM not initialized")
        raise Exception("SQL LLM not initialized")
    
    try:
        llm_logger.info("Generating SQL query using LLM...")
        prompt = ChatPromptTemplate.from_template(
            "{context}\n\nBased on the context above, write only the SQL query that answers the following question:\nQuestion: {question}\nSQL Query:"
        )
        chain = prompt | sql_llm
        response = chain.invoke({"context": context, "question": question})
        
        query = (
            str(response)
            .replace("SQL Query:", "")
            .replace("```sql", "")
            .replace("```", "")
            .strip()
            .strip('"')
        )
        
        llm_logger.info(f"Generated SQL query: {query}")
        return query
        
    except Exception as e:
        llm_logger.error(f"Error generating SQL query: {e}")
        raise

# --------------------- Data Summary ---------------------
def get_basic_data_summary(data):
    """Generate basic data summary"""
    logger.info("Generating basic data summary...")
    
    try:
        summary_lines = []
        summary_lines.append(f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns.")
        summary_lines.append(f"The dataset has the following columns: {', '.join(data.columns)}")
        
        missing = data.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            summary_lines.append("Missing values found:")
            for col, count in missing.items():
                summary_lines.append(f"  - {col}: {count} missing")

        numeric = data.select_dtypes(include='number')
        if not numeric.empty:
            desc = numeric.describe().T
            for col in desc.index:
                summary_lines.append(
                    f"{col}: min={desc.loc[col, 'min']}, mean={desc.loc[col, 'mean']:.2f}, max={desc.loc[col, 'max']}"
                )

        summary = "\n".join(summary_lines)
        logger.info("Basic data summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating basic data summary: {e}")
        return f"Error generating summary: {str(e)}"

def get_profiling_summary(data):
    """Get key insights from pandas profiling without generating full HTML"""
    logger.info("Generating profiling summary...")
    
    try:
        profile = ProfileReport(data, minimal=True)
        
        # Extract key statistics
        summary = {
            "overview": {
                "rows": data.shape[0],
                "columns": data.shape[1],
                "missing_cells": data.isnull().sum().sum(),
                "missing_percentage": (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            },
            "variables": {}
        }
        
        # Add variable-specific insights
        for col in data.columns:
            col_info = {
                "type": str(data[col].dtype),
                "missing_count": data[col].isnull().sum(),
                "unique_count": data[col].nunique()
            }
            
            if data[col].dtype in ['int64', 'float64']:
                col_info.update({
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max()
                })
            
            summary["variables"][col] = col_info
        
        logger.info("Profiling summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating profiling summary: {e}")
        print(f"Error generating profiling summary: {e}")
        return None

# --------------------- SQL EXECUTION WITH MAPPING ---------------------

def read_data_from_db(query):
    db_logger.info(f"Executing database query: {query}")
    
    try:
        db_logger.info("Establishing database connection...")
        db_logger.debug(f"Connecting to host: {os.getenv('DB_HOST')}")
        
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME")
        )
        
        db_logger.info("Database connection established successfully")
        

        cursor = conn.cursor()
        operation = query.strip().split()[0].upper()
        db_logger.info(f"Executing {operation} operation")

        if operation == "SELECT":
            db_logger.info("Executing SELECT query...")
            cursor.execute(query)

            if cursor.description is not None:
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                db_logger.info(f"Query returned {len(results)} rows with columns: {columns}")
        
                raw_data = [dict(zip(columns, row)) for row in results]
                transformed_data = header_mapper.transform_data_result(raw_data)
                output = {"data": transformed_data, "type": "select"}
        
                db_logger.info("SELECT query executed successfully")
            else:
                db_logger.warning("No columns returned in SELECT query.")
                output = {"data": [], "type": "select", "message": "No data returned."}


            
        else:
            db_logger.info(f"Executing {operation} query...")
            cursor.execute(query)
            conn.commit()
            row_count = cursor.rowcount
            
            db_logger.info(f"{operation} query affected {row_count} rows")

            if operation == "INSERT":
                message = f"✅ Successfully inserted {row_count} record(s)."
            elif operation == "UPDATE":
                message = f"🛠️ Successfully updated {row_count} record(s)."
            elif operation == "DELETE":
                message = f"🗑️ Successfully deleted {row_count} record(s)."
            elif operation == "ALTER":
                message = f"✅ Table structure altered successfully."
            else:
                message = f"✅ Query executed. {row_count} row(s) affected."

            output = {"data": message, "type": "non_select"}

        cursor.close()
        conn.close()
        db_logger.info("Database connection closed")

    except Exception as e:
        db_logger.error(f"Database error: {e}")
        output = {"data": f"❌ Error executing query: {e}", "type": "error"}

    return output

# --------------------- FLASK ROUTES ---------------------

@app.route('/')
def index():
    api_logger.info("Serving index page")
    return render_template('index.html')



@app.route('/get', methods=['POST'])
def chatbot_response():
    start_time = datetime.now()
    api_logger.info(f"Received chatbot request at {start_time}")
    
    user_msg = request.form['msg']
    api_logger.info(f"User message: {user_msg}")
    
    query = ""
    
    try:
        api_logger.info("meri jjjjjjaaaaaaaaannnnnnnnn")
        api_logger.info("Processing chatbot request...")
        
        # Step 1: Get the raw response from the LLM
        api_logger.info("Generating response from LLM...")
        llm_response = get_assistant_response(user_msg, context)
        api_logger.info(f"Generated response: {llm_response}")

        # Step 2: Check for special keywords from the LLM before executing a query
        response_upper = llm_response.strip().upper()

        # Check for greetings
        if response_upper == "GREETING":
            api_logger.info("Identified as a greeting. Sending friendly response.")
            return jsonify({
                "message": "Hello! I'm a database assistant. How can I help you with your contract data today?",
                "results": {"data": [], "type": "greeting"},
                
            })
        
        # Check for controlled errors from the LLM
        elif response_upper.startswith("ERROR:"):
            api_logger.warning(f"LLM returned a controlled error: {llm_response}")
            return jsonify({
                "message": llm_response, # Pass the LLM's error message directly to the user
                "results": {"data": [], "type": "llm_error"},
                
            })

        # Step 3: If no special keywords were found, it must be a query
        query = llm_response
        
        # Execute the database query
        api_logger.info("Executing database query...")
        result = read_data_from_db(query)
        api_logger.info(f"Query result type: {result.get('type')}")
        
        # Initialize response variables
        basic_summary = ""
        
        # Process results based on type
        if result.get("type") == "select" and isinstance(result.get("data"), list) and result["data"]:
            api_logger.info("Processing SELECT query results...")
            
            df = pd.DataFrame(result["data"])
            api_logger.info(f"Created DataFrame with shape: {df.shape}")
            
            basic_summary = get_basic_data_summary(df)
            api_logger.info("Basic summary generated")
            
        elif result.get("type") == "non_select":
            api_logger.info("Processing non-SELECT query results...")
            basic_summary = result["data"]
        else:
            api_logger.warning("Processing error or empty results...")
            basic_summary = result["data"]
        
        # Consistent response structure
        response_data = {
            "success": True,
            "message": basic_summary,
            "results": result
               # Add this line
            }

        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        api_logger.info(f"Request processed successfully in {processing_time:.2f} seconds")
        
        return jsonify(response_data)

    except Exception as e:
        api_logger.info("Meri jaaan  bbhen ki lodi")
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        api_logger.error(f"Request failed after {processing_time:.2f} seconds: {str(e)}")
        
        return jsonify({
            "message": f"Sorry, I encountered an error: {str(e)}",
            "sql_query": query,
            "results": {"data": f"Error: {str(e)}", "type": "error"},
            "profiling_summary": None
        })
    

# --------------------- APP START ---------------------

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    logger.info("Application running on http://localhost:5050")
    app.run(debug=True, port=5050)