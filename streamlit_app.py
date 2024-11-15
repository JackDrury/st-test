import streamlit as st
import sqlite3
import pandas as pd
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

model_version = "gpt-4o-mini"

# Database connection
@st.cache_resource
def init_connection():
    return sqlite3.connect('sales.db', check_same_thread=False)

conn = init_connection()

# Initialize database with sample data
def init_db():
    cursor = conn.cursor()
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            product_id INTEGER,
            quantity INTEGER,
            revenue FLOAT,
            customer_id INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            price FLOAT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            country TEXT
        )
    ''')
    
    # Insert sample data if tables are empty
    if not cursor.execute("SELECT * FROM products LIMIT 1").fetchone():
        cursor.executemany(
            "INSERT INTO products (name, category, price) VALUES (?, ?, ?)",
            [
                ("Laptop", "Electronics", 999.99),
                ("Smartphone", "Electronics", 699.99),
                ("Headphones", "Electronics", 149.99),
                ("Coffee Maker", "Appliances", 79.99),
                ("Blender", "Appliances", 49.99)
            ]
        )
        
        cursor.executemany(
            "INSERT INTO customers (name, email, country) VALUES (?, ?, ?)",
            [
                ("John Doe", "john@example.com", "USA"),
                ("Jane Smith", "jane@example.com", "Canada"),
                ("Alice Johnson", "alice@example.com", "UK"),
                ("Bob Wilson", "bob@example.com", "Australia"),
                ("Carol Brown", "carol@example.com", "USA")
            ]
        )
        
        # Insert sample sales data
        import random
        from datetime import datetime, timedelta
        
        for i in range(100):
            date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
            product_id = random.randint(1, 5)
            quantity = random.randint(1, 5)
            price = cursor.execute("SELECT price FROM products WHERE id=?", (product_id,)).fetchone()[0]
            revenue = price * quantity
            customer_id = random.randint(1, 5)
            
            cursor.execute("""
                INSERT INTO sales (date, product_id, quantity, revenue, customer_id)
                VALUES (?, ?, ?, ?, ?)
            """, (date.date(), product_id, quantity, revenue, customer_id))
    
    conn.commit()

# Initialize the database
init_db()

# Function to get table schema
def get_schema_description():
    cursor = conn.cursor()
    schema = []
    
    # Get tables and their schemas
    for table in ['sales', 'products', 'customers']:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        schema.append(f"Table '{table}' with columns: " + 
                     ", ".join([f"{col[1]} ({col[2]})" for col in columns]))
    
    return "\n".join(schema)

# Function to generate SQL query using OpenAI
def generate_sql_query(prompt, schema):
    system_message = f"""You are a SQL expert. Generate a SQLite query based on the user's request.
The database has the following schema:

{schema}

Return ONLY the SQL query, no explanations or additional text. The query should be valid SQLite syntax."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model_version,
        messages=messages,
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

# Function to execute SQL query safely
def execute_query(query):
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

# Streamlit UI
st.title('📊 SQL Query Generator')
st.markdown('Use natural language to query your database!')

# Sidebar with database schema
with st.sidebar:
    st.header('Database Schema')
    st.text(get_schema_description())
    
    st.header('Sample Questions')
    st.markdown("""
    Try asking questions like:
    - Show me total revenue by product category
    - Who are the top 5 customers by revenue?
    - What's the monthly sales trend?
    """)

# Main query interface
query_prompt = st.text_area('What would you like to know about the data?', 
                           height=100,
                           placeholder="e.g., Show me total revenue by product category")

col1, col2 = st.columns([1, 1])

with col1:
    generate_button = st.button('Generate Query')
    
if generate_button and query_prompt:
    # Generate SQL query
#    sql_query = generate_sql_query(query_prompt, get_schema_description())
############################################################################
    schema = get_schema_description()
    prompt = f"""
        You are a SQL expert. There is a database with the following schema:
        {schema}
        Now please convert the question below into working SQL and execute it:
        {query_prompt}
    """
    messages = [{"role": "user", "content": prompt}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_query",
                "description": "Execute the given SQL query and return the results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_query": {
                            "type": "string",
                            "description": "The SQL query to execute"
                        }                
                    },
                    "required": ["target_query"]
                }
            }

        }
    ]


    response = client.chat.completions.create(
        model=model_version,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    print("=============================")
    print(f'here is the first gpt response message:')
#    logging.info(f'gpt_response1:{response_message}')
    print(response_message)
    print("=============================")

# Step 2: check if GPT wanted to call a function
    tool_calls = response_message.tool_calls
    if tool_calls:
# Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors

        available_functions = {
            "execute_query": execute_query,
        }  # only one function in this example, but you can have multiple

        messages.append(response_message)
        
        list_of_queries = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            target_q = function_args.get("target_query") # This is the query that the agent decided it wants to use
            list_of_queries.append(target_q)
            function_response = function_to_call(
                target_query=target_q 
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response


        second_response = client.chat.completions.create(
            model=model_version,
            messages=messages,
        )  # get a new response from the model where it can see the function response
#        print("=============================")
#        print("second (final) response:")
#        print(second_response)
#        logging.info(f'gpt_response2:{second_response}')
#        print('=============================END=============================')
#        return second_response

#
#    else:
#        print("apparently GPT didn't want to call a function....")


############################################################################
    
        # Display the generated queries
        st.code('\n'.join(list_of_queries), language='sql')
    
    # Show result of LAST QUERY (note that the below assumes one query while above allows many, please rectify)
        if function_response is not None:
            st.header('Query Results')
            st.subheader('Explanation of what is going on')
            st.text(second_response)
        
        # Display results
            st.dataframe(results)
        
        # Show visualization options if applicable
            if len(results) > 0 and len(results.columns) >= 2:
                st.header('Visualization')
            
            # Detect numeric columns
                numeric_cols = results.select_dtypes(include=['float64', 'int64']).columns
            
                if len(numeric_cols) > 0:
                    chart_type = st.selectbox('Select chart type:', 
                                            ['bar', 'line', 'scatter'])
                
                    if chart_type == 'bar':
                        st.bar_chart(results)
                    elif chart_type == 'line':
                        st.line_chart(results)
                    else:
                        st.scatter_chart(results)

# Add export functionality
#if 'results' in locals() and results is not None:
#    st.download_button(
##        label="Download results as CSV",
#        data=results.to_csv(index=False).encode('utf-8'),
#        file_name="query_results.csv",
#        mime="text/csv"
    )

# Cleanup connection when app is done
def cleanup():
    conn.close()

st.cache_resource.clear()
