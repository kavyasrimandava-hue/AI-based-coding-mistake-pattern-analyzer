import ast
import os
import time
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Based Coding Mistake Pattern Analyzer",
    page_icon="💻",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# GEMINI CLIENT
# ============================================================

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


# ============================================================
# DATABASE
# ============================================================

DB_NAME = "mistakes.db"


def create_database():
    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mistakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mistake_type TEXT NOT NULL,
            description TEXT NOT NULL,
            code TEXT NOT NULL,
            line_number INTEGER,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_mistake(
    mistake_type,
    description,
    code,
    line_number=None
):
    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO mistakes
        (
            mistake_type,
            description,
            code,
            line_number,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        mistake_type,
        description,
        code,
        line_number,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_mistake_history():
    conn = sqlite3.connect(DB_NAME)

    query = """
        SELECT
            id,
            mistake_type,
            description,
            code,
            line_number,
            created_at
        FROM mistakes
        ORDER BY id DESC
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


def clear_mistake_history():
    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("DELETE FROM mistakes")

    conn.commit()
    conn.close()


create_database()


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

if "mistake_counts" not in st.session_state:
    st.session_state.mistake_counts = {
        "Syntax Error": 0,
        "Undefined Variable": 0,
        "Unused Variable": 0,
        "Division by Zero": 0
    }


# ============================================================
# ML TRAINING DATA - 100 EXAMPLES
# 25 EXAMPLES FOR EACH CLASS
# ============================================================

training_data = [

    # ========================================================
    # 1. SYNTAX ERROR - 25 EXAMPLES
    # ========================================================

    (
        "print('Hello'",
        "Closing parenthesis is missing",
        "Syntax Error"
    ),

    (
        "if x > 10",
        "Missing colon in if statement",
        "Syntax Error"
    ),

    (
        "for i in range(10",
        "Closing parenthesis is missing",
        "Syntax Error"
    ),

    (
        "def hello(",
        "Incomplete function definition",
        "Syntax Error"
    ),

    (
        "name = 'Kavya",
        "Unclosed string",
        "Syntax Error"
    ),

    (
        "numbers = [1, 2, 3",
        "List is not closed",
        "Syntax Error"
    ),

    (
        "print('Welcome'",
        "Missing closing parenthesis",
        "Syntax Error"
    ),

    (
        "if age > 18",
        "Missing colon after if condition",
        "Syntax Error"
    ),

    (
        "def calculate(x, y)",
        "Missing colon in function definition",
        "Syntax Error"
    ),

    (
        "for i in range(5:",
        "Invalid parenthesis syntax",
        "Syntax Error"
    ),

    (
        "print('Good morning",
        "String is not closed",
        "Syntax Error"
    ),

    (
        "values = [10, 20, 30",
        "Missing closing square bracket",
        "Syntax Error"
    ),

    (
        "while x < 10",
        "Missing colon in while statement",
        "Syntax Error"
    ),

    (
        "def add(a, b)",
        "Missing colon after function definition",
        "Syntax Error"
    ),

    (
        "student = {'name': 'Kavya'",
        "Dictionary is not closed",
        "Syntax Error"
    ),

    (
        "print((10 + 20)",
        "Closing parenthesis is missing",
        "Syntax Error"
    ),

    (
        "if marks >= 40",
        "Missing colon after condition",
        "Syntax Error"
    ),

    (
        "for item in items",
        "Missing colon in for loop",
        "Syntax Error"
    ),

    (
        "def greet(name)",
        "Missing colon in function definition",
        "Syntax Error"
    ),

    (
        "message = \"Hello",
        "Unclosed string literal",
        "Syntax Error"
    ),

    (
        "numbers = (1, 2, 3",
        "Tuple is not closed",
        "Syntax Error"
    ),

    (
        "if x == 5",
        "Missing colon in if statement",
        "Syntax Error"
    ),

    (
        "while count < 5",
        "Missing colon in while loop",
        "Syntax Error"
    ),

    (
        "print([1, 2, 3)",
        "Mismatched brackets",
        "Syntax Error"
    ),

    (
        "def multiply(a, b",
        "Function parameter list is incomplete",
        "Syntax Error"
    ),


    # ========================================================
    # 2. UNDEFINED VARIABLE - 25 EXAMPLES
    # ========================================================

    (
        "print(name)",
        "Variable name is not defined",
        "Undefined Variable"
    ),

    (
        "print(age)",
        "Variable age is not defined",
        "Undefined Variable"
    ),

    (
        "print(student_name)",
        "Variable student_name is not defined",
        "Undefined Variable"
    ),

    (
        "result = marks + 10",
        "Variable marks is not defined",
        "Undefined Variable"
    ),

    (
        "print(total)",
        "Variable total is not defined",
        "Undefined Variable"
    ),

    (
        "print(score)",
        "Variable score is not defined",
        "Undefined Variable"
    ),

    (
        "print(username)",
        "Variable username is not defined",
        "Undefined Variable"
    ),

    (
        "value = unknown + 5",
        "Variable unknown is not defined",
        "Undefined Variable"
    ),

    (
        "print(city)",
        "Variable city is not defined",
        "Undefined Variable"
    ),

    (
        "total = price + tax",
        "Variables price and tax are not defined",
        "Undefined Variable"
    ),

    (
        "average = total / count",
        "Variables total and count are not defined",
        "Undefined Variable"
    ),

    (
        "print(student)",
        "Variable student is not defined",
        "Undefined Variable"
    ),

    (
        "result = x + y",
        "Variables x and y are not defined",
        "Undefined Variable"
    ),

    (
        "print(mark)",
        "Variable mark is not defined",
        "Undefined Variable"
    ),

    (
        "salary = basic + bonus",
        "Variables basic and bonus are not defined",
        "Undefined Variable"
    ),

    (
        "print(subject)",
        "Variable subject is not defined",
        "Undefined Variable"
    ),

    (
        "total = amount + fee",
        "Variables amount and fee are not defined",
        "Undefined Variable"
    ),

    (
        "print(country)",
        "Variable country is not defined",
        "Undefined Variable"
    ),

    (
        "result = first + second",
        "Variables first and second are not defined",
        "Undefined Variable"
    ),

    (
        "print(department)",
        "Variable department is not defined",
        "Undefined Variable"
    ),

    (
        "average = marks / students",
        "Variables marks and students are not defined",
        "Undefined Variable"
    ),

    (
        "print(phone)",
        "Variable phone is not defined",
        "Undefined Variable"
    ),

    (
        "total = salary + allowance",
        "Variables salary and allowance are not defined",
        "Undefined Variable"
    ),

    (
        "print(address)",
        "Variable address is not defined",
        "Undefined Variable"
    ),

    (
        "result = quantity * price",
        "Variables quantity and price are not defined",
        "Undefined Variable"
    ),


    # ========================================================
    # 3. UNUSED VARIABLE - 25 EXAMPLES
    # ========================================================

    (
        "name = 'Kavya'",
        "Variable name is defined but never used",
        "Unused Variable"
    ),

    (
        "age = 19",
        "Variable age is defined but never used",
        "Unused Variable"
    ),

    (
        "marks = 90",
        "Variable marks is defined but never used",
        "Unused Variable"
    ),

    (
        "student = 'Kavya'",
        "Variable student is defined but never used",
        "Unused Variable"
    ),

    (
        "result = 100",
        "Variable result is defined but never used",
        "Unused Variable"
    ),

    (
        "total = 500",
        "Variable total is defined but never used",
        "Unused Variable"
    ),

    (
        "salary = 50000",
        "Variable salary is defined but never used",
        "Unused Variable"
    ),

    (
        "count = 10",
        "Variable count is defined but never used",
        "Unused Variable"
    ),

    (
        "city = 'Hyderabad'",
        "Variable city is defined but never used",
        "Unused Variable"
    ),

    (
        "price = 1000",
        "Variable price is defined but never used",
        "Unused Variable"
    ),

    (
        "percentage = 85",
        "Variable percentage is defined but never used",
        "Unused Variable"
    ),

    (
        "student_id = 101",
        "Variable student_id is defined but never used",
        "Unused Variable"
    ),

    (
        "temperature = 30",
        "Variable temperature is defined but never used",
        "Unused Variable"
    ),

    (
        "number = 50",
        "Variable number is defined but never used",
        "Unused Variable"
    ),

    (
        "value = 200",
        "Variable value is defined but never used",
        "Unused Variable"
    ),

    (
        "address = 'Hyderabad'",
        "Variable address is defined but never used",
        "Unused Variable"
    ),

    (
        "phone = 9876543210",
        "Variable phone is defined but never used",
        "Unused Variable"
    ),

    (
        "subject = 'Python'",
        "Variable subject is defined but never used",
        "Unused Variable"
    ),

    (
        "score = 95",
        "Variable score is defined but never used",
        "Unused Variable"
    ),

    (
        "height = 160",
        "Variable height is defined but never used",
        "Unused Variable"
    ),

    (
        "weight = 58",
        "Variable weight is defined but never used",
        "Unused Variable"
    ),

    (
        "course = 'AIML'",
        "Variable course is defined but never used",
        "Unused Variable"
    ),

    (
        "college = 'MRECW'",
        "Variable college is defined but never used",
        "Unused Variable"
    ),

    (
        "experience = 2",
        "Variable experience is defined but never used",
        "Unused Variable"
    ),

    (
        "percentage = 90",
        "Variable percentage is defined but never used",
        "Unused Variable"
    ),


    # ========================================================
    # 4. DIVISION BY ZERO - 25 EXAMPLES
    # ========================================================

    (
        "result = 10 / 0",
        "Division by zero",
        "Division by Zero"
    ),

    (
        "a = 10\nb = 0\nresult = a / b",
        "Division by zero because b is zero",
        "Division by Zero"
    ),

    (
        "x = 20\nresult = x // 0",
        "Floor division by zero",
        "Division by Zero"
    ),

    (
        "number = 10\nresult = number % 0",
        "Modulo by zero",
        "Division by Zero"
    ),

    (
        "value = 100 / 0",
        "Division by zero",
        "Division by Zero"
    ),

    (
        "a = 50\nb = 0\nprint(a / b)",
        "Division by zero because b is zero",
        "Division by Zero"
    ),

    (
        "total = 20\ncount = 0\naverage = total / count",
        "Division by zero because count is zero",
        "Division by Zero"
    ),

    (
        "x = 5\ny = 0\nz = x % y",
        "Modulo by zero",
        "Division by Zero"
    ),

    (
        "marks = 100\nstudents = 0\naverage = marks / students",
        "Division by zero because students is zero",
        "Division by Zero"
    ),

    (
        "total = 500\ncount = 0\nresult = total / count",
        "Division by zero because count is zero",
        "Division by Zero"
    ),

    (
        "a = 20\nb = 0\nc = a // b",
        "Floor division by zero",
        "Division by Zero"
    ),

    (
        "x = 10\ny = 0\nanswer = x / y",
        "Division by zero",
        "Division by Zero"
    ),

    (
        "amount = 100\nquantity = 0\nprice = amount / quantity",
        "Division by zero because quantity is zero",
        "Division by Zero"
    ),

    (
        "numerator = 50\ndenominator = 0\nresult = numerator / denominator",
        "Division by zero because denominator is zero",
        "Division by Zero"
    ),

    (
        "total = 100\nparts = 0\nvalue = total % parts",
        "Modulo by zero",
        "Division by Zero"
    ),

    (
        "x = 40\ny = 0\nresult = x / y",
        "Division by zero because y is zero",
        "Division by Zero"
    ),

    (
        "distance = 100\ntime = 0\nspeed = distance / time",
        "Division by zero because time is zero",
        "Division by Zero"
    ),

    (
        "score = 90\nstudents = 0\naverage = score / students",
        "Division by zero because students is zero",
        "Division by Zero"
    ),

    (
        "amount = 500\npeople = 0\nshare = amount / people",
        "Division by zero because people is zero",
        "Division by Zero"
    ),

    (
        "number = 25\nzero = 0\nresult = number // zero",
        "Floor division by zero",
        "Division by Zero"
    ),

    (
        "value = 30\nzero = 0\nresult = value % zero",
        "Modulo by zero",
        "Division by Zero"
    ),

    (
        "total = 1000\ndivisor = 0\nanswer = total / divisor",
        "Division by zero because divisor is zero",
        "Division by Zero"
    ),

    (
        "profit = 200\nmonths = 0\naverage = profit / months",
        "Division by zero because months is zero",
        "Division by Zero"
    ),

    (
        "distance = 500\nhours = 0\nspeed = distance / hours",
        "Division by zero because hours is zero",
        "Division by Zero"
    ),

    (
        "numerator = 80\ndenominator = 0\nanswer = numerator / denominator",
        "Division by zero because denominator is zero",
        "Division by Zero"
    )
]


# ============================================================
# PREPARE ML DATA
# ============================================================

training_text = [
    code + " " + description
    for code, description, label in training_data
]

training_labels = [
    label
    for code, description, label in training_data
]


# ============================================================
# MAIN ML MODEL
# ============================================================

ml_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000
        )
    )
])


ml_model.fit(
    training_text,
    training_labels
)


# ============================================================
# ML PREDICTION FUNCTION
# ============================================================

def predict_mistake(code, description=""):

    text = code + " " + description

    prediction = ml_model.predict([text])[0]

    probabilities = ml_model.predict_proba([text])[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# ============================================================
# CODE ANALYSIS FUNCTIONS
# ============================================================

def find_undefined_variables(tree):

    defined_variables = set()
    used_variables = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Name):

            if isinstance(node.ctx, ast.Store):

                defined_variables.add(node.id)

            elif isinstance(node.ctx, ast.Load):

                used_variables.append(
                    (node.id, node.lineno)
                )

    builtin_names = {
        "print",
        "len",
        "range",
        "int",
        "float",
        "str",
        "list",
        "dict",
        "set",
        "tuple",
        "sum",
        "max",
        "min",
        "abs",
        "input",
        "True",
        "False",
        "None"
    }

    undefined = []

    for variable, line in used_variables:

        if (
            variable not in defined_variables
            and variable not in builtin_names
        ):

            undefined.append(
                (variable, line)
            )

    return undefined


def find_unused_variables(tree):

    assigned = {}
    used = set()

    for node in ast.walk(tree):

        if isinstance(node, ast.Name):

            if isinstance(node.ctx, ast.Store):

                assigned[node.id] = node.lineno

            elif isinstance(node.ctx, ast.Load):

                used.add(node.id)

    unused = []

    for variable, line in assigned.items():

        if variable not in used:

            unused.append(
                (variable, line)
            )

    return unused


def find_division_by_zero(tree):

    constant_values = {}

    results = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Assign):

            if len(node.targets) == 1:

                target = node.targets[0]

                if isinstance(target, ast.Name):

                    try:

                        value = ast.literal_eval(node.value)

                        constant_values[target.id] = value

                    except Exception:

                        pass

        if isinstance(node, ast.BinOp):

            if isinstance(
                node.op,
                (ast.Div, ast.FloorDiv, ast.Mod)
            ):

                divisor = node.right

                is_zero = False

                if isinstance(
                    divisor,
                    ast.Constant
                ):

                    if divisor.value == 0:

                        is_zero = True

                elif isinstance(
                    divisor,
                    ast.Name
                ):

                    variable = divisor.id

                    if (
                        variable in constant_values
                        and constant_values[variable] == 0
                    ):

                        is_zero = True

                if is_zero:

                    results.append(
                        node.lineno
                    )

    return results


def analyze_code(code):

    mistakes = []

    try:

        tree = ast.parse(code)

    except SyntaxError as e:

        mistakes.append({
            "type": "Syntax Error",
            "description": e.msg,
            "line": e.lineno,
            "column": e.offset
        })

        return mistakes

    undefined = find_undefined_variables(tree)

    for variable, line in undefined:

        mistakes.append({
            "type": "Undefined Variable",
            "description":
                f"Variable `{variable}` may be undefined.",
            "line": line
        })

    unused = find_unused_variables(tree)

    for variable, line in unused:

        mistakes.append({
            "type": "Unused Variable",
            "description":
                f"Variable `{variable}` is defined on line {line} but is never used.",
            "line": line
        })

    division_errors = find_division_by_zero(tree)

    for line in division_errors:

        mistakes.append({
            "type": "Division by Zero",
            "description":
                f"Division by zero detected on line {line}.",
            "line": line
        })

    return mistakes


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def get_gemini_explanation(code, mistakes):

    if not client:

        return "Gemini API key was not found."

    mistake_text = "\n".join(
        [
            f"- {m['type']}: {m['description']}"
            for m in mistakes
        ]
    )

    prompt = f"""
You are an expert Python programming tutor.

Analyze the following Python code.

CODE:
{code}

DETECTED MISTAKES:
{mistake_text}

Explain:

1. What is wrong?
2. Why does the mistake happen?
3. How can it be fixed?
4. Show corrected code.

Keep the explanation simple and beginner-friendly.
"""

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            return response.text

        except Exception as e:

            error_text = str(e)

            if "503" in error_text:

                if attempt < 2:

                    time.sleep(3)

                    continue

            return f"Gemini Error: {e}"

    return "Unable to get Gemini explanation."


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🔍 Code Analyzer",
        "📊 Mistake Patterns",
        "🕒 Mistake History",
        "🤖 ML Classification",
        "📈 ML Evaluation",
        "📊 Dashboard"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "💻 AI-Based Coding Mistake Pattern Analyzer"
    )

    st.write(
        """
        This application analyzes Python code,
        detects common coding mistakes,
        identifies mistake patterns,
        and uses Machine Learning and Gemini AI
        to provide additional insights.
        """
    )

    st.subheader("✨ Features")

    st.markdown("""
    - 🔍 Syntax Error Detection
    - ❓ Undefined Variable Detection
    - ⚠️ Unused Variable Detection
    - ➗ Division by Zero Detection
    - 🤖 Machine Learning Classification
    - 📈 ML Model Evaluation
    - 🧠 Gemini AI Explanation
    - 🕒 Mistake History
    - 📊 Dashboard
    """)


# ============================================================
# CODE ANALYZER
# ============================================================

elif page == "🔍 Code Analyzer":

    st.title("🔍 Code Analyzer")

    code = st.text_area(
        "Enter your Python code:",
        height=300,
        placeholder="Write your Python code here..."
    )

    analyze_button = st.button(
        "🔍 Analyze Code"
    )

    if analyze_button:

        if not code.strip():

            st.warning(
                "Please enter some Python code."
            )

        else:

            st.session_state.analysis_count += 1

            mistakes = analyze_code(code)

            if not mistakes:

                st.success(
                    "✅ No basic coding mistakes detected!"
                )

            else:

                for mistake in mistakes:

                    mistake_type = mistake["type"]

                    description = mistake["description"]

                    line = mistake.get("line")

                    st.session_state.mistake_counts[
                        mistake_type
                    ] += 1

                    save_mistake(
                        mistake_type,
                        description,
                        code,
                        line
                    )

                    if mistake_type == "Syntax Error":

                        st.error(
                            "❌ Syntax Error Detected!"
                        )

                    elif mistake_type == "Undefined Variable":

                        st.warning(
                            "❓ Possible Undefined Variable!"
                        )

                    elif mistake_type == "Unused Variable":

                        st.warning(
                            "⚠️ Unused Variables Found!"
                        )

                    elif mistake_type == "Division by Zero":

                        st.error(
                            "➗ Possible Division by Zero!"
                        )

                    st.write(description)

                    if line:

                        st.write(
                            f"Line: {line}"
                        )

                st.divider()

                st.subheader(
                    "🤖 Gemini AI Explanation"
                )

                with st.spinner(
                    "Gemini is analyzing your code..."
                ):

                    explanation = get_gemini_explanation(
                        code,
                        mistakes
                    )

                st.markdown(explanation)

                st.divider()

                st.subheader(
                    "🤖 ML Classification"
                )

                primary_description = mistakes[0]["description"]

                prediction, confidence = predict_mistake(
                    code,
                    primary_description
                )

                st.write(
                    f"**Predicted Mistake:** {prediction}"
                )

                st.write(
                    f"**Model Confidence:** {confidence:.2f}%"
                )


# ============================================================
# MISTAKE PATTERNS
# ============================================================

elif page == "📊 Mistake Patterns":

    st.title("📊 Mistake Patterns")

    counts = st.session_state.mistake_counts

    data = pd.DataFrame(
        {
            "Mistake Type": list(counts.keys()),
            "Count": list(counts.values())
        }
    )

    st.bar_chart(
        data.set_index("Mistake Type")
    )

    st.subheader(
        "Current Session Statistics"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Analyses",
            st.session_state.analysis_count
        )

    with col2:

        st.metric(
            "Total Mistakes",
            sum(counts.values())
        )


# ============================================================
# MISTAKE HISTORY
# ============================================================

elif page == "🕒 Mistake History":

    st.title("🕒 Mistake History")

    history = get_mistake_history()

    if history.empty:

        st.info(
            "No mistake history available yet."
        )

    else:

        st.dataframe(
            history,
            use_container_width=True
        )

        st.subheader(
            "Mistake Type Summary"
        )

        summary = (
            history["mistake_type"]
            .value_counts()
            .reset_index()
        )

        summary.columns = [
            "Mistake Type",
            "Count"
        ]

        st.dataframe(
            summary,
            use_container_width=True
        )

        if st.button(
            "🗑️ Clear Mistake History"
        ):

            clear_mistake_history()

            st.success(
                "Mistake history cleared."
            )

            st.rerun()


# ============================================================
# ML CLASSIFICATION
# ============================================================

elif page == "🤖 ML Classification":

    st.title("🤖 ML Classification")

    st.write(
        """
        Enter Python code and let the Machine Learning
        model predict the most likely mistake type.
        """
    )

    ml_code = st.text_area(
        "Enter code for ML prediction:",
        height=250,
        key="ml_code"
    )

    ml_description = st.text_input(
        "Optional mistake description:",
        key="ml_description"
    )

    if st.button(
        "🤖 Predict Mistake"
    ):

        if not ml_code.strip():

            st.warning(
                "Please enter some code."
            )

        else:

            prediction, confidence = predict_mistake(
                ml_code,
                ml_description
            )

            st.success(
                f"Predicted Mistake: {prediction}"
            )

            st.info(
                f"Model Confidence: {confidence:.2f}%"
            )

            st.caption(
                "This is a prototype ML model trained on a small hand-curated dataset."
            )


# ============================================================
# ML EVALUATION
# ============================================================

elif page == "📈 ML Evaluation":

    st.title("📈 ML Model Evaluation")

    st.write(
        """
        This section evaluates the Machine Learning model
        using a separate test portion of the labeled dataset.
        """
    )

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        training_text,
        training_labels,
        test_size=0.25,
        random_state=42,
        stratify=training_labels
    )

    # --------------------------------------------------------
    # EVALUATION MODEL
    # --------------------------------------------------------

    evaluation_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000
            )
        )
    ])

    evaluation_model.fit(
        X_train,
        y_train
    )

    y_pred = evaluation_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # DISPLAY SAMPLE COUNTS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Examples",
            len(training_data)
        )

    with col2:

        st.metric(
            "Training Examples",
            len(X_train)
        )

    with col3:

        st.metric(
            "Testing Examples",
            len(X_test)
        )

    st.divider()

    # --------------------------------------------------------
    # DISPLAY METRICS
    # --------------------------------------------------------

    st.subheader(
        "📊 Performance Metrics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🔢 Confusion Matrix"
    )

    class_names = [
        "Syntax Error",
        "Undefined Variable",
        "Unused Variable",
        "Division by Zero"
    ]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_names
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.caption(
        "Rows represent the actual mistake class, while columns represent the predicted class."
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📋 Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.info(
        """
        ⚠️ Note: This model is a prototype trained on a
        hand-curated dataset. These evaluation results are
        an estimate of performance on this particular test
        split, not production-level accuracy.
        """
    )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.title("📊 Dashboard")

    history = get_mistake_history()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Analyses",
            st.session_state.analysis_count
        )

    with col2:

        st.metric(
            "Total Detected Mistakes",
            sum(
                st.session_state.mistake_counts.values()
            )
        )

    with col3:

        st.metric(
            "Stored Mistakes",
            len(history)
        )

    st.divider()

    st.subheader(
        "Mistake Distribution"
    )

    if history.empty:

        st.info(
            "No stored mistakes yet."
        )

    else:

        chart_data = (
            history["mistake_type"]
            .value_counts()
        )

        st.bar_chart(
            chart_data
        )

    st.divider()

    st.subheader(
        "Current Session Mistakes"
    )

    session_df = pd.DataFrame(
        {
            "Mistake Type":
                list(
                    st.session_state.mistake_counts.keys()
                ),

            "Count":
                list(
                    st.session_state.mistake_counts.values()
                )
        }
    )

    st.dataframe(
        session_df,
        use_container_width=True
    )