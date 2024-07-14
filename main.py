from langchain_groq import ChatGroq
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from fewShots import fewShots

import os
os.environ["PINECONE_API_KEY"] = "c3b7c476-25f1-48e3-bb6e-c3039d406cc1"

def get_few_shot_db_chain():
    db_user = "sql12719782"
    db_password = "3epgTLwYRt"
    db_host = "sql12.freesqldatabase.com"
    db_name = "sql12719782"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)
    llm = ChatGroq(
        temperature=0.7,
        model="llama3-70b-8192",
        api_key="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"
    )

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in fewShots]
    vectorstore = PineconeVectorStore.from_texts(to_vectorize, embeddings, metadatas=fewShots, index_name="zaki-index")


    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2, #select 2 similar
    )

    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )


    chain = create_sql_query_chain(llm, db, prompt=few_shot_prompt)
    return chain, db

def main():

    st.title("Backend SQL-Generative-Bot using Retrieval Augmented Generation")

    question = st.text_input("Question: ")

    if question:
        chain, db = get_few_shot_db_chain()
        response = chain.invoke({"question":question})
        st.header("Answer")
        answer = db._execute(response.split("SQLQuery")[-1][1:])
        st.write(answer)
        print(db.table_info)
        


if __name__ == "__main__":
    main()
