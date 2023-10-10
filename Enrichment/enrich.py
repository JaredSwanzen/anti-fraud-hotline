import pandas as pd

# Make sure to add the data iin the relavatn path.
# NEEDS A LOT OF CLEANINGGGGGGGGG, Some things are null, false spelt wrong,
pd.set_option('display.max_columns',None)
file_paths = "./data/"

data_merchant = pd.read_csv(f"{file_paths}merchants.csv")
data_terminal = pd.read_csv(f"{file_paths}terminals.csv")
data_transactions = pd.read_csv(f"{file_paths}transactions_train.csv", low_memory=False)
data_customer = pd.read_csv(f"{file_paths}customers.csv")

def get_heads():
    print(data_transactions.head())
    print(data_customer.describe())
    print(data_terminal.describe())
    print(data_merchant.describe())

def enrich():
    enriched = data_transactions.join(data_customer.set_index("CUSTOMER_ID"), on="CUSTOMER_ID", how='inner')
    enriched = enriched.join(data_terminal.set_index('TERMINAL_ID'), on="TERMINAL_ID", how='inner')
    enriched.to_csv('./no_merchant.csv')
    enriched = enriched.join(data_merchant.set_index('MERCHANT_ID'), on='MERCHANT_ID', how='inner')
    enriched.to_csv('./enriched.csv')
    print(enriched.describe())

if __name__ == '__main__':
    get_heads()
    print("Now enriching")
    enrich()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
