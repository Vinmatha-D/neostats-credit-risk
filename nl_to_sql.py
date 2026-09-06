"""
Natural Language to SQL Query Module
Converts NL questions to SQL queries using Claude LLM
"""

import os
import sqlite3
import re
from typing import Tuple, List, Dict
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Database schema (simplified for credit risk dataset)
DB_SCHEMA = """
CREATE TABLE applicants (
    SK_ID_CURR INT PRIMARY KEY,
    TARGET INT,  -- 0: No default, 1: Default
    AGE INT,
    INCOME INT,
    ANNUITY_AMOUNT INT,
    CREDIT_AMOUNT INT,
    DEBT_AMOUNT INT,
    EMPLOYMENT_YEARS INT,
    CREDIT_SCORE INT,
    EDUCATION TEXT,
    FAMILY_STATUS TEXT,
    HOUSING_TYPE TEXT,
    OCCUPATION_TYPE TEXT,
    INCOME_TYPE TEXT,
    ORGANIZATION_TYPE TEXT,
    DAYS_EMPLOYED INT,
    LOAN_TO_VALUE_RATIO FLOAT,
    DEBT_TO_INCOME_RATIO FLOAT
);
"""

# Few-shot examples for prompt engineering
FEW_SHOT_EXAMPLES = """
Examples of natural language → SQL conversions:

Q: "What's the average income by education level?"
A: SELECT education, ROUND(AVG(income), 2) as avg_income FROM applicants GROUP BY education ORDER BY avg_income DESC;

Q: "How many applicants defaulted in the age 30-40 bracket?"
A: SELECT COUNT(*) as count, SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) as defaulted FROM applicants WHERE age BETWEEN 30 AND 40;

Q: "Top 10 riskiest applicants by debt-to-income ratio?"
A: SELECT sk_id_curr, debt_to_income_ratio, credit_score FROM applicants ORDER BY debt_to_income_ratio DESC LIMIT 10;

Q: "Default rate by employment years?"
A: SELECT 
    CASE WHEN employment_years < 1 THEN '<1'
         WHEN employment_years < 5 THEN '1-5'
         ELSE '>5' END as tenure,
    COUNT(*) as total,
    ROUND(100.0 * SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct
FROM applicants
GROUP BY tenure
ORDER BY tenure;

Q: "Which occupation type has the highest default rate?"
A: SELECT occupation_type, COUNT(*) as total, SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) as defaults,
    ROUND(100.0 * SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct
FROM applicants GROUP BY occupation_type ORDER BY default_rate_pct DESC LIMIT 5;
"""


def nl_to_sql_with_claude(question: str) -> str:
    """
    Convert natural language question to SQL using Claude API
    
    Args:
        question: User's natural language question
    
    Returns:
        SQL query string
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    prompt = f"""You are a SQL expert for credit risk analysis database.
Convert natural language questions to SQL queries.

Database Schema:
{DB_SCHEMA}

{FEW_SHOT_EXAMPLES}

RULES:
1. Only use SELECT queries (no INSERT, UPDATE, DELETE, DROP)
2. Return ONLY the SQL query, no explanation or markdown
3. Ensure all table and column names are lowercase
4. Use LIMIT 20 for safety (unless user specifies otherwise)
5. Always use ROUND() for percentages with 2 decimal places
6. Never use functions not in SQLite standard library
7. If ambiguous, make reasonable assumptions

User Question: {question}

SQL Query:"""
    
    message = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=300,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    sql_query = message.content[0].text.strip()
    
    # Clean up if wrapped in backticks
    if sql_query.startswith("```"):
        sql_query = sql_query.split("```")[1]
        if sql_query.startswith("sql"):
            sql_query = sql_query[3:]
    
    sql_query = sql_query.strip()
    
    return sql_query


def validate_sql_safety(sql_query: str) -> Tuple[bool, str]:
    """
    Validate SQL query for safety (no DROP, DELETE, INSERT, etc.)
    
    Args:
        sql_query: SQL query to validate
    
    Returns:
        (is_safe, message)
    """
    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'TRUNCATE']
    
    # Check for dangerous keywords
    query_upper = sql_query.upper()
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False, f"❌ Query contains dangerous keyword: {keyword}"
    
    # Check if it's a SELECT query
    if not query_upper.strip().startswith('SELECT'):
        return False, "❌ Only SELECT queries are allowed"
    
    return True, "✅ Query is safe to execute"


def execute_sql_query(sql_query: str, db_path: str = "data/credit_data.db") -> Tuple[bool, str]:
    """
    Execute SQL query and return results
    
    Args:
        sql_query: SQL query to execute
        db_path: Path to SQLite database
    
    Returns:
        (success, result_string)
    """
    # Validate query
    is_safe, safety_msg = validate_sql_safety(sql_query)
    if not is_safe:
        return False, safety_msg
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return True, "No results found."
        
        # Get column names
        col_names = [description[0] for description in cursor.description]
        
        # Format results as markdown table
        result = "| " + " | ".join(col_names) + " |\n"
        result += "|" + "|".join(["---"] * len(col_names)) + "|\n"
        
        # Add rows (limit to 20)
        for row in rows[:20]:
            formatted_row = []
            for val in row:
                if isinstance(val, float):
                    formatted_row.append(f"{val:.2f}")
                else:
                    formatted_row.append(str(val) if val is not None else "NULL")
            result += "| " + " | ".join(formatted_row) + " |\n"
        
        if len(rows) > 20:
            result += f"\n*(Showing 20 of {len(rows)} results)*"
        
        conn.close()
        return True, result
    
    except sqlite3.Error as e:
        return False, f"❌ Database error: {str(e)}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def chat_with_data(user_question: str) -> Dict[str, str]:
    """
    Full pipeline: NL question → SQL → Results
    
    Args:
        user_question: Natural language question from user
    
    Returns:
        Dictionary with question, sql, and result
    """
    print(f"\n📝 User Question: {user_question}")
    
    # Step 1: Convert NL to SQL
    print("🔄 Converting to SQL...")
    sql_query = nl_to_sql_with_claude(user_question)
    print(f"✅ SQL Generated:\n{sql_query}")
    
    # Step 2: Execute query
    print("⚡ Executing query...")
    success, result = execute_sql_query(sql_query)
    
    if success:
        print(f"✅ Query executed successfully!")
    else:
        print(f"❌ Query execution failed: {result}")
    
    return {
        "question": user_question,
        "sql_query": sql_query,
        "result": result,
        "success": success
    }


# ============= HARDCODED FALLBACK QUERIES =============
# Use these if LLM fails or network issues

FALLBACK_QUERIES = {
    "default_rate": {
        "query": "SELECT COUNT(*) as total, SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) as defaults, ROUND(100.0 * SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct FROM applicants;",
        "keywords": ["default rate", "overall default"]
    },
    "age_default": {
        "query": "SELECT CASE WHEN age < 25 THEN '<25' WHEN age < 35 THEN '25-35' WHEN age < 50 THEN '35-50' ELSE '>50' END as age_group, COUNT(*) as total, ROUND(100.0 * SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct FROM applicants GROUP BY age_group ORDER BY age_group;",
        "keywords": ["age", "default"]
    },
    "income_default": {
        "query": "SELECT CASE WHEN income < 150000 THEN '<1.5L' WHEN income < 250000 THEN '1.5-2.5L' WHEN income < 500000 THEN '2.5-5L' ELSE '>5L' END as income_bracket, COUNT(*) as total, ROUND(100.0 * SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct FROM applicants GROUP BY income_bracket ORDER BY income_bracket;",
        "keywords": ["income", "default"]
    },
    "employment_default": {
        "query": "SELECT CASE WHEN employment_years < 1 THEN '<1' WHEN employment_years < 5 THEN '1-5' ELSE '>5' END as tenure, COUNT(*) as total, ROUND(100.0 * SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct FROM applicants GROUP BY tenure ORDER BY tenure;",
        "keywords": ["employment", "years", "default"]
    },
    "top_riskiest": {
        "query": "SELECT sk_id_curr, debt_to_income_ratio, credit_score, income FROM applicants ORDER BY debt_to_income_ratio DESC LIMIT 10;",
        "keywords": ["riskiest", "top", "debt"]
    }
}


def find_fallback_query(question: str) -> str:
    """
    Find a fallback query if LLM fails
    """
    question_lower = question.lower()
    
    for query_key, query_data in FALLBACK_QUERIES.items():
        if any(keyword in question_lower for keyword in query_data['keywords']):
            return query_data['query']
    
    # Default: return all applicants
    return "SELECT * FROM applicants LIMIT 5;"


# ============= MAIN ENTRY POINT =============

if __name__ == "__main__":
    # Example usage
    test_questions = [
        "What's the default rate by age group?",
        "How many applicants defaulted?",
        "Top 10 riskiest applicants by debt-to-income?",
    ]
    
    for question in test_questions:
        result = chat_with_data(question)
        print(f"\n{'='*60}")
        print(f"Question: {result['question']}")
        print(f"SQL: {result['sql_query']}")
        print(f"Success: {result['success']}")
        print(f"Result:\n{result['result']}")
