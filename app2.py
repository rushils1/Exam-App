import streamlit as st
import pandas as pd
import sqlite3
import google.generativeai as genai

# Configure Google Gemini API
genai.configure(api_key="AIzaSyCbjsFoBKfTkEnTNRvudYbdtkCTqUAO_as")  # Replace with your actual API Key

# Function to preprocess uploaded student performance data
def preprocess_data(file):
    if file is None:
        st.error("No file uploaded.")
        st.stop()

    if file.size == 0:
        st.error("The uploaded file is empty. Please upload a valid file.")
        st.stop()

    file.seek(0)  # Reset file pointer

    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, sep=None, engine="python", header=0)
        else:
            df = pd.read_excel(file, header=0)
    except pd.errors.EmptyDataError:
        st.error("The file is empty or has no readable columns.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

    # Fix unnamed columns
    updated_columns = []
    prev_subject = None

    for col in df.columns:
        if "Unnamed" in col and prev_subject:
            updated_columns.append(prev_subject)
        else:
            updated_columns.append(col)
            prev_subject = col

    df.columns = updated_columns

    # Merge first row into headers where applicable
    static_headers = {
        "Add.ID", "Student No", "Student Name", "Campus", "GPA", "CGPA", "Status",
        "Progression Status", "Total Marks", "Marks scored", "Percentage", "Curr.Sem.F",
        "Curr.Year.F", "Cum.Total.F", "Prev.Tot.F", "Total attempted credits", "Remarks"
    }

    first_row = df.iloc[0].fillna("")  # Replace NaNs with empty strings

    df.columns = [
        f"{col} {first_row[i]}".strip() if col not in static_headers else col
        for i, col in enumerate(df.columns)
    ]

    # Remove first row after merging headers
    df = df.iloc[2:].reset_index(drop=True)

    return df

# Streamlit UI
st.title("🎓 Student Performance Analysis Chatbot 🤖")

uploaded_file = st.file_uploader("📂 Upload Student Data File", type=["csv", "xlsx"])

if uploaded_file is not None:
    raw_df = preprocess_data(uploaded_file)

    if raw_df is not None:
        st.write("### 📋 Cleaned Data Preview:")
        st.dataframe(raw_df.head())

        st.success("✅ Data is processed! You can now query it.")

        # Extract metadata for AI model (Column names)
        column_names = ", ".join(raw_df.columns)

        # Identify dynamic subject-related columns
        subject_columns = [col for col in raw_df.columns if "Marks" in col or "Grade" in col]

        if not subject_columns:
            st.error("No subject columns found in the dataset.")
            st.stop()

        # Few-shot learning examples to fine-tune AI
        few_shot_examples = f"""
        Example 1:
        User: "Which campus has the highest average GPA?"
        SQL Query: SELECT Campus, AVG(GPA) as avg_gpa FROM student_performance GROUP BY Campus ORDER BY avg_gpa DESC LIMIT 1;

        Example 2:
        User: "Who are the top 5 students with the highest CGPA?"
        SQL Query: SELECT `Student Name`, CGPA FROM student_performance ORDER BY CGPA DESC LIMIT 5;

        Example 3:
        User: "What is the pass percentage in [Subject Name]?"
        SQL Query: SELECT CAST(SUM(CASE WHEN `{subject_columns[0]}` NOT IN ('F', 'AB') THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) AS Pass_Percentage FROM student_performance;

        Example 4:
        User: "How many students passed in [Subject Name]?"
        SQL Query: SELECT CAST(SUM(CASE WHEN `{subject_columns[0]}` NOT IN ('F', 'AB') THEN 1 ELSE 0 END) AS REAL) AS Pass_Count FROM student_performance;

        Example 5:
        User: "How many students failed in [Subject Name]?"
        SQL Query: SELECT CAST(SUM(CASE WHEN `{subject_columns[0]}` IN ('F', 'AB') THEN 1 ELSE 0 END) AS REAL) AS Fail_Count FROM student_performance;
        """

        # System prompt including dataset metadata
        system_prompt = f"""
        You are an expert SQLite assistant. Answer only using valid SQLite queries based on the student performance database.
        
        Dataset contains the following columns:
        {column_names}

        Rules:
        - Do NOT include backticks (`) or SQL code blocks (```sql).
        - Generate only the SQL query.
        - Always match column names exactly as they appear in the dataset.
        - Handle dynamic subject names and select the most appropriate column.

        Few-shot examples:
        {few_shot_examples}
        """

        # Initialize AI chat session
        chat = genai.GenerativeModel("gemini-2.0-flash").start_chat()
        chat.send_message(system_prompt, generation_config={"temperature": 0})

        st.write("### 💬 Chat with the AI")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask me anything about student performance...")

        if user_input:
            # Ask AI for the SQL query
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            response = chat.send_message(user_input + " Strictly do NOT include backticks (`) or SQL code blocks (```sql).")
            sql_query = response.text.strip()

            # Handle dynamic subject mapping
            for subject in subject_columns:
                if any(word in user_input.lower() for word in subject.lower().split()):
                    sql_query = sql_query.replace("{subject_column}", subject)
                    break

            # Execute SQL query safely with a new SQLite connection each time
            try:
                with sqlite3.connect(":memory:") as conn:
                    raw_df.to_sql("student_performance", conn, index=False, if_exists="replace")  # Recreate in-memory DB
                    result_df = pd.read_sql_query(sql_query, conn)

                # Format result into a natural response
                if not result_df.empty:
                    if "Pass_Percentage" in result_df.columns:
                        result_value = result_df.iloc[0, 0]
                        result_text = f"📊 The pass percentage in {subject} is **{result_value:.2f}%**."
                    elif "avg_gpa" in result_df.columns:
                        result_text = f"🏆 The campus with the highest average GPA is **{result_df.iloc[0,0]}** with an average GPA of **{result_df.iloc[0,1]:.2f}**."
                    else:
                        result_text = f"✅ Here’s what I found:\n\n{result_df.to_markdown(index=False)}"
                else:
                    if "poorly" in user_input.lower() or "fail" in user_input.lower() or "low marks" in user_input.lower():
                        result_text = "✅ No one has performed poorly in this subject. 🎉"
                    else:
                        result_text = "❌ No results found. Provide a better prompt."

            except Exception as e:
                result_text = f"❌ Error executing query: {str(e)}"

            st.session_state.chat_history.append({"role": "assistant", "content": result_text})

            with st.chat_message("assistant"):
                st.markdown(result_text)
