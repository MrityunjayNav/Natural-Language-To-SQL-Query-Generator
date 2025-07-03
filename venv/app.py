import mysql.connector
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from ydata_profiling import ProfileReport
import tempfile
import json

app = Flask(__name__)
load_dotenv()

# --------------------- COLUMN DISPLAY MAPPING ---------------------

COLUMN_MAPPINGS = {
    'patient_id': 'Patient ID',
    'first_name': 'First Name',
    'last_name': 'Last Name',
    'gender': 'Gender',
    'birth_date': 'Date of Birth',
    'city': 'City',
    'province_id': 'Province ID',
    'allergies': 'Allergies',
    'height': 'Height (cm)',
    'weight': 'Weight (kg)',
    'province_name': 'Province Name',
    'admission_date': 'Admission Date',
    'discharge_date': 'Discharge Date',
    'diagnosis': 'Diagnosis',
    'attending_doctor_id': 'Attending Doctor ID',
    'doctor_id': 'Doctor ID',
    'specialty': 'Specialty',
    'full_name': 'Full Name',
    'age': 'Age',
    'admission_duration': 'Days Admitted',
    'total_patients': 'Total Patients',
    'avg_age': 'Average Age',
    'patient_count': 'Patient Count'
}

class HeaderMapper:
    def __init__(self, mappings=None):
        self.mappings = mappings or COLUMN_MAPPINGS

    def get_display_name(self, column_name):
        return self.mappings.get(column_name, self._format_column_name(column_name))

    def _format_column_name(self, column_name):
        return ' '.join(word.capitalize() for word in column_name.split('_'))

    def transform_data_result(self, data_list):
        if not data_list:
            return []
        transformed_data = []
        for row in data_list:
            transformed_row = {
                self.get_display_name(k): v for k, v in row.items()
            }
            transformed_data.append(transformed_row)
        return transformed_data

# Initialize the header mapper and LLMs
header_mapper = HeaderMapper()

# Initialize LLMs once to avoid recreating on every request
try:
    sql_llm = OllamaLLM(model="mistral:7b-instruct", temperature=0.01)
except Exception as e:
    print(f"Error initializing LLMs: {e}")
    sql_llm = None

# --------------------- LLM CONTEXT & HELPER ---------------------

context = '''
You are a SQL query generator for MySQL database 'tut1'.

Tables:
- patients: patient_id, first_name, last_name, gender, birth_date, city, province_id, allergies, height, weight
- admissions: patient_id, admission_date, discharge_date, diagnosis, attending_doctor_id
- doctors: doctor_id, first_name, last_name, specialty  
- province_names: province_id, province_name

CRITICAL SCHEMA RULES:
- Gender column stores SINGLE CHARACTERS ONLY:
  * 'M' = male (NOT 'male')
  * 'F' = female (NOT 'female')
- Case sensitive: must use uppercase 'M' and 'F'
-Gender values are only 'M' or 'F'
-Specialty values in doctors table: 'Cardiologist', 'Internist', 'General Surgeon', 'Obstetrician/Gynecologist',"Gastroenterologist", etc.

STRICT RULES:
1. For "What are the different/unique [column]" questions → use SELECT DISTINCT
2. For "Find patients with [specific allergy]" → use LIKE with actual value
3. Return ONLY executable SQL, no placeholders like %keyword% . NO prefixes like "SQL:" or "Query:"
4. NO quotes around the statement
5. For names, search in first_name and/or last_name columns.Always use table aliases or full table.column names when joining tables.NEVER use COUNT() in WHERE clause - use HAVING instead
6. Always use proper JOINs when referencing multiple tables
7. For age calculations: DATEDIFF(CURDATE(), birth_date) / 365 > age_threshold
8. When columns exist in multiple tables, specify which table to use

EXAMPLES:
Question: "What are the different allergies patients have?"
SQL: SELECT DISTINCT allergies FROM patients;

Question: "Who are the cardiologists?"
SQL: SELECT * FROM doctors WHERE specialty = 'Cardiologist';

Question: "Find patients with penicillin allergy"  
SQL: SELECT * FROM patients WHERE allergies LIKE '%penicillin%';

Question: "What are the different cities?"
SQL: SELECT DISTINCT city FROM patients;

Question: "List all doctor specialties"
SQL: SELECT DISTINCT specialty FROM doctors;


'''

def get_assistant_response(question, context):
    if not sql_llm:
        raise Exception("SQL LLM not initialized")
    
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
    return query


# --------------------- Data Summary ---------------------
def get_basic_data_summary(data):
    """Generate basic data summary"""
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

    return "\n".join(summary_lines)

def get_profiling_summary(data):
    """Get key insights from pandas profiling without generating full HTML"""
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
            
        return summary
        
    except Exception as e:
        print(f"Error generating profiling summary: {e}")
        return None

# --------------------- SQL EXECUTION WITH MAPPING ---------------------

def read_data_from_db(query):
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        cursor = conn.cursor()
        operation = query.strip().split()[0].upper()

        if operation == "SELECT":
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            raw_data = [dict(zip(columns, row)) for row in results]
            # Transform headers
            transformed_data = header_mapper.transform_data_result(raw_data)
            output = {"data": transformed_data, "type": "select"}
        else:
            cursor.execute(query)
            conn.commit()
            row_count = cursor.rowcount

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

    except Exception as e:
        output = {"data": f"❌ Error executing query: {e}", "type": "error"}

    return output

# --------------------- FLASK ROUTES ---------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    user_msg = request.form['msg']
    query = ""
    
    try:
        # Generate SQL query
        query = get_assistant_response(user_msg, context)
        
        # Execute query
        result = read_data_from_db(query)
        
        # Initialize response variables
        basic_summary = ""
        
        # Process results based on type
        if result.get("type") == "select" and isinstance(result.get("data"), list) and result["data"]:
            # Create DataFrame for analysis
            df = pd.DataFrame(result["data"])
            
            # Generate basic summary
            basic_summary = get_basic_data_summary(df)
            

            
        elif result.get("type") == "non_select":
            basic_summary = result["data"]  # The success message
        else:
            basic_summary = result["data"]  # Error message
        
        # Consistent response structure
        response_data = {
            "message": basic_summary,
            "results": result,
            "query":query
        }
        
        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "message": f"Sorry, I encountered an error: {str(e)}",
            "sql_query": query,
            "results": {"data": f"Error: {str(e)}", "type": "error"},
            "profiling_summary": None,
            "profiling_report_path": None
        })

# --------------------- APP START ---------------------

if __name__ == "__main__":
    app.run(debug=True)