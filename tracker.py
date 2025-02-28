# tracker.py

def transactions():
    global transactions_list
    transactions_list = []

def add_transaction(transaction_type, amount, category, description):
    transactions_list.append({
        "Type": transaction_type,
        "Amount": amount,
        "Category": category,
        "Description": description
    })

def get_balance():
    balance = 0
    for transaction in transactions_list:
        if transaction["Type"] == "Income":
            balance += transaction["Amount"]
        elif transaction["Type"] == "Expense":
            balance -= transaction["Amount"]
    return balance

def get_expenses():
    return [t for t in transactions_list if t["Type"] == "Expense"]

def get_category_expenses():
    import pandas as pd
    df = pd.DataFrame(get_expenses())
    if not df.empty:
        return df.groupby("Category")["Amount"].sum()
    return None