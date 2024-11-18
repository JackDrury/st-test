import streamlit as st
import sqlite3
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

model_version = "gpt-4o-mini-2024-07-18"

# Database connection
@st.cache_resource
def init_connection():
    return sqlite3.connect('sales.db', check_same_thread=False)

#conn = init_connection()
conn = sqlite3.connect('boxing.db')

# Function to get table schema
def get_schema_description():
    cursor = conn.cursor()
    schema = []
    
    # Get tables and their schemas
    for table in ['bouts', 'champions', 'reigns', 'titles']:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        schema.append(f"Table '{table}' with columns: " + 
                     ", ".join([f"{col[1]} ({col[2]})" for col in columns]))
    
    return "\n\n".join(schema)

# Function to generate SQL query using OpenAI
def generate_sql_query(prompt, schema):
    system_message = f"""You are a SQL expert. Generate a SQLite query based on the user's request. The query you generate MUST be valid SQLite syntax.
The database has the following schema:

{schema}

Return ONLY the SQLite query, no explanations or additional text. The query MUST be valid SQLite syntax."""

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
    if query[-1] != ";":
        query += ";"

    cursor = conn.cursor() 

    cursor.execute(query)

    answer = cursor.fetchall()

    answer = json.dumps(answer)

    cursor.close()
    return answer
    
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
                    "required": ["target_query"],
                    "additionalProperties": False
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
    print("checking for tool calls")
    if tool_calls:
        print("entered tool calls loop")
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
            target_query = function_args.get("target_query") # This is the query that the agent decided it wants to use
            print(f"this is the target_query:\n{target_query}")
            list_of_queries.append(target_query)
            function_response = function_to_call(
                target_query 
            )
            print(f"this is function response:\n{function_response}")
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
        results = function_response
    # Show result of LAST QUERY (note that the below assumes one query while above allows many, please rectify)
        if function_response is not None:
            st.header('Query Results')
            st.subheader('Explanation of what is going on')
            st.text(second_response.choices[0].message.content)
        
        # Display results
#            st.dataframe(results)
        
        # Show visualization options if applicable
#            if len(results) > 0 and len(results.columns) >= 2:
#                st.header('Visualization')
            
            # Detect numeric columns
#                numeric_cols = results.select_dtypes(include=['float64', 'int64']).columns
            
#                if len(numeric_cols) > 0:
#                    chart_type = st.selectbox('Select chart type:', 
#                                            ['bar', 'line', 'scatter'])
                
#                    if chart_type == 'bar':
#                        st.bar_chart(results)
#                    elif chart_type == 'line':
#                        st.line_chart(results)
#                    else:
#                        st.scatter_chart(results)

# Add export functionality
#if 'results' in locals() and results is not None:
#    st.download_button(
##        label="Download results as CSV",
#        data=results.to_csv(index=False).encode('utf-8'),
#        file_name="query_results.csv",
#        mime="text/csv"
#    )

# Cleanup connection when app is done
def cleanup():
    conn.close()

st.cache_resource.clear()
