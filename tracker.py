# tracker.py

# ✅ Initialize transactions_list globally
transactions_list = []

def add_transaction(transaction_type, amount, category, description):
    global transactions_list  # ✅ Ensure it modifies the global list
    transactions_list.append({
        "Type": transaction_type,
        "Amount": amount,
        "Category": category,
        "Description": description
    })

def get_balance():
    global transactions_list  # ✅ Ensure it accesses the global list
    balance = 0
    for transaction in transactions_list:
        if transaction["Type"] == "Income":
            balance += transaction["Amount"]
        elif transaction["Type"] == "Expense":
            balance -= transaction["Amount"]
    return balance

# def get_expenses():
#     global transactions_list  # ✅ Ensure it accesses the global list
#     if not isinstance(transactions_list, list):
#         return []
#     return [t for t in transactions_list if t.get("Type") == "Expense"]

# def get_category_expenses():
#     import pandas as pd
#     df = pd.DataFrame(get_expenses())
#     if not df.empty:
#         return df.groupby("Category")["Amount"].sum()
#     return None
