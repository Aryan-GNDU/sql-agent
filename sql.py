import os
import getpass
from typing_extensions import TypedDict, Annotated
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from IPython.display import display, Image
from langchain.prompts import SystemMessagePromptTemplate

# ✅ Connect to MySQL Database
try:
    db = SQLDatabase.from_uri("mysql+pymysql://root:root@localhost/aryandb1")
    print("Connected to:", db.dialect)
    print("Tables:", db.get_usable_table_names())
except Exception as e:
    print(f"Database connection error: {e}")
    exit()

# ✅ Define State Structure
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# ✅ Secure API Key Input
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# ✅ Initialize Llama 3 Model from Groq
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# ✅ Custom SQL Query Prompt (Better Column Handling)
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

query_prompt_template.messages[0] = SystemMessagePromptTemplate.from_template(
    query_prompt_template.messages[0].prompt.template + """
    
Special Query Rules:
1. If the question includes "song", search in `Song_Name` column.
2. If the question includes "artist" or "singer", search in `Artist` column.
3. Use LOWER() in MySQL to make searches case-insensitive.
4. Use LIMIT {top_k} unless a specific number is requested.

Example Queries:
- "How many songs are there?" ➝ `SELECT COUNT(DISTINCT Song_Name) FROM music_dataset;`
- "Show top 5 songs of Coldplay" ➝ `SELECT Song_Name FROM music_dataset WHERE LOWER(Artist) = LOWER('Coldplay') LIMIT 5;`
- "How many songs are there?" ➝ SELECT COUNT(DISTINCT Song_Name) FROM music_dataset;
- "Show top 5 songs of Coldplay" ➝ SELECT Song_Name FROM music_dataset WHERE LOWER(Artist) = LOWER('Coldplay') LIMIT 5;
"""
)

# ✅ Define SQL Query Output Schema
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# ✅ Generate SQL Query
def write_query(state: State, max_retries=3):
    """Generate SQL query to fetch information based on user question with retry logic."""
    print("write_query state:", state)
    attempt = 0
    sql_query = ""
    
    while attempt < max_retries:
        try:
            # Optionally, add extra instruction on retries to help guide the query generation.
            extra_instruction = ""
            if attempt > 0:
                extra_instruction = " Please ensure you use the correct column mapping for the query."
                
            prompt = query_prompt_template.invoke(
                {
                    "dialect": db.dialect,
                    "top_k": 10,
                    "table_info": db.get_table_info(),
                    "input": state["question"] + extra_instruction,
                }
            )
            
            structured_llm = llm.with_structured_output(QueryOutput)
            result = structured_llm.invoke(prompt)
            sql_query = result["query"]
            
            # Basic validation: Check that the output contains a SELECT statement.
            if sql_query and "SELECT" in sql_query.upper():
                print(f"Valid query generated on attempt {attempt+1}: {sql_query}")
                break
            else:
                print(f"Attempt {attempt+1} returned an invalid query: {sql_query}. Retrying...")
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}. Retrying...")
            
        attempt += 1

    if not sql_query or "SELECT" not in sql_query.upper():
        raise ValueError("Failed to generate a valid SQL query after multiple attempts.")
    
    return {"query": sql_query}

# ✅ Execute SQL Query with Error Handling
def execute_query(state: State):
    """Execute the generated SQL query and return results."""
    try:
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        result = execute_query_tool.invoke(state["query"])
        return {"result": result}
    except Exception as e:
        return {"result": f"Query execution failed: {str(e)}"}

# ✅ Generate Answer from SQL Query Result
def generate_answer(state: State):
    """Generate a natural language response from SQL query results."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# ✅ Build Execution Graph
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

# ✅ Display Graph
display(Image(graph.get_graph().draw_mermaid_png()))


# ✅ Function to Ask Query in Terminal
def ask_query():
    """Ask a query from the user in the terminal and process it."""
    user_question = input("\nEnter your query: ")  # Take input from the user
    print("\nProcessing query... 🔄")

    state = {"question": user_question}

    for step in graph.stream(state, stream_mode="updates"):
        print("Step output:", step)  # Debug print for every step
        if "query" in step:
            print("Generated SQL Query: 📝", step["query"])
        elif "result" in step:
            print("Query Results: 📊", step["result"])
        elif "answer" in step:
            print("Final Answer: ✅", step["answer"])


# ✅ Run Query Pipeline from Terminal
if __name__ == "__main__":
    while True:
        ask_query()
        cont = input("\nDo you want to ask another question? (yes/no): ").strip().lower()
        if cont != "yes":
            print("Goodbye! 👋")
            break
# Test query generation separately
test_state = {"question": "How many songs does Coldplay have?"}
generated = write_query(test_state)
print("Generated query:", generated["query"])
