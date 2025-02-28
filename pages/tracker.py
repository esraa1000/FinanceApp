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


def get_expenses():
    global transactions_list
    return [t for t in transactions_list if t.get("Type") == "Expense"]

def plot_doughnut_chart(data, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'}, pctdistance=0.85)
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)
    ax.set_title(title)
    return fig

def get_category_expenses():
    import pandas as pd
    df = pd.DataFrame(get_expenses())
    if not df.empty:
        return df.groupby("Category")["Amount"].sum()
    return None