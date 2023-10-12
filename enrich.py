import pandas as pd
import numpy as np
import pytz

# Make sure to add the data iin the relavatn path.
# NEEDS A LOT OF CLEANINGGGGGGGGG, Some things are null, false spelt wrong,
pd.set_option('display.max_columns',None)
file_paths = "./data/"

dtype = {"IS_RECURRING_TRANSACTION": str}
data_merchant = pd.read_csv(f"{file_paths}merchants.csv")
data_terminal = pd.read_csv(f"{file_paths}terminals.csv")
data_transactions = pd.read_csv(f"{file_paths}transactions_train.csv", dtype=dtype)
data_customer = pd.read_csv(f"{file_paths}customers.csv")

def combine():
    # combined = data_transactions.join(data_customer.set_index("CUSTOMER_ID"), on="CUSTOMER_ID", how='inner')
    # combined = combined.join(data_terminal.set_index('TERMINAL_ID'), on="TERMINAL_ID", how='inner')
    # combined = combined.join(data_merchant.set_index('MERCHANT_ID'), on='MERCHANT_ID', how='inner')

    # print(combined.size)
    # combined = combined[combined['IS_RECURRING_TRANSACTION'] != 'Fals']
    # print(combined.size)

    # combined.to_csv(f'{file_paths}combined.csv', index=False)

    combined = pd.read_csv(f"{file_paths}combined.csv")
    enrich(combined)

def enrich(df):
    # TODO: enrich with EXPIRED_CARD, TRANSACTION_TO_EXPIRY_DISTANCE, CUSTOMER_TO_TERMINAL_DISTANCE, TRANSACTION_TOTAL, TRANSACTION_FAILED

    df['CARD_EXPIRY_DATE'] = pd.to_datetime(df['CARD_EXPIRY_DATE'], format='%m/%y').dt.tz_localize(pytz.UTC)
    df['TX_TS'] = pd.to_datetime(df['TX_TS'])
    df['TRANSACTION_TO_EXPIRY_DISTANCE'] = (df['CARD_EXPIRY_DATE'] - df['TX_TS']).dt.days
    print(df[['TRANSACTION_TO_EXPIRY_DISTANCE', 'TX_FRAUD']].corr())

    df['CUSTOMER_TO_TERMINAL_DISTANCE'] = np.sqrt((df['x_customer_id'] - df['x_terminal_id'])**2 + (df['y_customer_id'] - df['y_terminal__id'])**2)
    print(df[['CUSTOMER_TO_TERMINAL_DISTANCE', 'TX_FRAUD']].corr())

    df['TRANSACTION_TOTAL'] = df['TX_AMOUNT'] - df['TRANSACTION_CASHBACK_AMOUNT']
    print(df[['TRANSACTION_TOTAL', 'TX_FRAUD']].corr())

    df['TRANSACTION_FAILED'] = df['FAILURE_CODE'].notna()
    print(df[['TRANSACTION_FAILED', 'TX_FRAUD']].corr())

    df.to_csv(f'{file_paths}enriched.csv', index=False)

if __name__ == '__main__':
    # get_heads()
    print("enriching")
    
    combine()
