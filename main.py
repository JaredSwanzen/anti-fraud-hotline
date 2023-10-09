import os
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


excluded_columns = ['TX_FRAUD', 'TX_ID', 'TX_TS', 'CUSTOMER_ID', 'TERMINAL_ID', 'CARD_DATA', 'MERCHANT_ID', 'ACQUIRER_ID']

df = pd.read_csv('data/transactions_train.csv')

categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
fraudulent_data = df[df['TX_FRAUD'] == 1]
non_fraudulent_data = df[df['TX_FRAUD'] == 0]

# Create a directory to save the graphs (optional)
os.makedirs('output_graphs', exist_ok=True)

# Set the directory where you want to save the graphs
output_directory = 'output_graphs'

# Histograms or Kernel Density Plots for Numerical Features
for column in df.columns:
    if column not in excluded_columns and column not in categorical_columns:
        plt.figure(figsize=(14, 8))
        sns.kdeplot(fraudulent_data[column], label='Fraudulent', fill=True)
        sns.kdeplot(non_fraudulent_data[column], label='Non-Fraudulent', fill=True)
        plt.title(f'Distribution of {column} by Fraud Status')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()

        # Save the plot to a file
        plt.savefig(f'{output_directory}/{column}_distribution.png')
        plt.close()

# Barplots for Categorical Features
for column in categorical_columns:
    if column not in excluded_columns:
        if column == 'FAILURE_REASON':
          plt.figure(figsize=(20, 14))
          ax = sns.countplot(y=column, data=df, hue='TX_FRAUD')
          plt.ylabel(column)
          plt.xlabel('Count')
          for p in ax.patches:
            count = int(p.get_height())  # Get the count for the current bar
            x = p.get_x() + p.get_width() / 2.  # X-coordinate of the bar's center
            y = p.get_height()  # Y-coordinate at the top of the bar
            ax.annotate(f'{count}', (x, y), ha='center', va='bottom')

        else:
          plt.figure(figsize=(14, 8))
          ax = sns.countplot(x=column, data=df, hue='TX_FRAUD')
          plt.ylabel('Count')
          plt.xlabel(column)
          for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')


        plt.title(f'{column} Distribution by Fraud Status')
        plt.legend(title='TX_FRAUD', labels=['Non-Fraudulent', 'Fraudulent'])
        
        # Save the plot to a file
        plt.savefig(f'{output_directory}/{column}_distribution.png')
        plt.close()
