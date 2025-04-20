import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import re

# ✅ Set Streamlit config first (very important!)
st.set_page_config(page_title="💸 AI Daily Expense Tracker", page_icon="📊", layout="centered")

# Load HuggingFace zero-shot-classification model
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

# Define labels
CATEGORY_LABELS = ["Food", "Transport", "Utilities", "Entertainment", "Shopping", "Healthcare", "Others"]

# Session state for tracking entries
if "expenses" not in st.session_state:
    st.session_state.expenses = []

# --- Title and Input Section ---
st.title("💸 AI + Manual Expense Tracker")
st.markdown("Track your daily expenses — manually or using AI! Categorize, visualize, and stay in budget.")

st.sidebar.header("📝 Manual Input")

# --- Manual Input via Sidebar ---
categories_input = st.sidebar.text_area("Enter categories (comma-separated):", "Food, Transport, Entertainment")
amounts_input = st.sidebar.text_area("Enter corresponding amounts (comma-separated):", "500, 200, 150")

# --- Optional Budget Input ---
user_budget = st.sidebar.number_input("💰 Set Your Budget (optional):", value=1000.0, step=100.0)

# --- Add from Manual Input ---
if st.sidebar.button("➕ Add Manual Expenses"):
    categories = [cat.strip() for cat in categories_input.split(',')]
    try:
        amounts = [float(amount.strip()) for amount in amounts_input.split(',')]
    except ValueError:
        st.sidebar.error("Amounts must be numeric.")
        amounts = []

    if len(categories) != len(amounts):
        st.sidebar.error("Number of categories and amounts must match.")
    else:
        for cat, amt in zip(categories, amounts):
            st.session_state.expenses.append({
                "Description": f"Manual: {cat}",
                "Amount": amt,
                "Category": cat
            })
        st.sidebar.success("✅ Manual expenses added!")

# --- AI Section ---
st.markdown("### 🤖 Enter your expense in plain text (e.g., 'Paid ₹600 for medicines')")

user_input = st.text_input("AI Expense Entry:")

def extract_amount(text):
    match = re.search(r"(?:₹|INR)?\s?(\d+[.,]?\d*)", text)
    return float(match.group(1)) if match else 0.0

def classify_category(text):
    result = classifier(text, candidate_labels=CATEGORY_LABELS)
    return result["labels"][0] if result else "Others"

if st.button("➕ Add AI-Predicted Expense") and user_input:
    amount = extract_amount(user_input)
    category = classify_category(user_input)

    st.session_state.expenses.append({
        "Description": user_input,
        "Amount": amount,
        "Category": category
    })
    st.success(f"Added ₹{amount:.2f} under '{category}'")

# --- Clear All Expenses (Delete Old) ---
if st.button("🗑️ Clear All Expenses Before Adding New"):
    st.session_state.expenses = []
    st.success("✅ All previous expenses have been deleted. You can add new expenses now!")

# --- Expense Table ---
if st.session_state.expenses:
    st.subheader("📋 Expense Log")
    df = pd.DataFrame(st.session_state.expenses)
    st.dataframe(df, use_container_width=True)

    total = df["Amount"].sum()
    st.metric("🧾 Total Spent", f"₹{total:.2f}")

    # --- Budget Alert ---
    if user_budget > 0:
        if total > user_budget:
            st.warning(f"⚠️ Budget Exceeded! Your budget was ₹{user_budget:.2f}, but you've spent ₹{total:.2f}")
        else:
            st.success(f"🎉 You're within your budget of ₹{user_budget:.2f}!")

    # --- Summary Chart ---
    st.subheader("📊 Summary by Category")
    summary = df.groupby("Category")["Amount"].sum().reset_index().sort_values(by="Amount", ascending=False)

    # Pie Chart
    fig, ax = plt.subplots()
    ax.pie(summary["Amount"], labels=summary["Category"], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Bar Chart
    st.bar_chart(summary.set_index("Category"))

    # --- Summary Report ---
    st.subheader("📝 Expense Summary")
    avg_expense = total / len(df)
    top_category = summary.iloc[0]["Category"]
    top_amount = summary.iloc[0]["Amount"]

    st.text_area("AI Summary", f"""
Total expenses: ₹{total:.2f}
Average expense per entry: ₹{avg_expense:.2f}
Top category: {top_category} — ₹{top_amount:.2f}
""", height=150)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Ruchika Srivatava")

