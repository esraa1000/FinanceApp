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
    global transactions_list

    # Ensure transactions_list is initialized
    if not isinstance(transactions_list, list):
        print("Error: transactions_list is not a list or is None. Initializing it.")
        transactions_list = []

    # Debugging output
    print("Transactions List:", transactions_list)

    # Filter expenses safely
    return [t for t in transactions_list if t.get("Type") == "Expense"]


def get_category_expenses():
    import pandas as pd
    df = pd.DataFrame(get_expenses())
    if not df.empty:
        return df.groupby("Category")["Amount"].sum()
    return None