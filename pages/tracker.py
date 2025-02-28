transactions_list = []

def add_transaction(transaction_type, amount, category, description):
    global transactions_list
    transactions_list.append({
        "Type": transaction_type,
        "Amount": amount,
        "Category": category,
        "Description": description
    })

def get_balance():
    global transactions_list
    balance = 0
    for transaction in transactions_list:
        if transaction["Type"] == "Income":
            balance += transaction["Amount"]
        elif transaction["Type"] == "Expense":
            balance -= transaction["Amount"]
    return balance

# ✅ Uncommented and corrected get_expenses()
def get_expenses():
    global transactions_list
    return [t for t in transactions_list if t.get("Type") == "Expense"]