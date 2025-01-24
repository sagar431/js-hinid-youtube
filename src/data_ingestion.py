import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

try:
    # Read data from URL
    df = pd.read_csv('https://raw.githubusercontent.com/araj2/customer-database/master/Ecommerce%20Customers.csv')

    # Select columns from index 3 onwards
    df = df.iloc[:, 3:]

    # Filter rows where Length of Membership > 3
    df = df[df['Length of Membership'] > 3]

    # Drop specified column
    df.drop(columns=['Avg. Session Length'], inplace=True)

    # Save processed data
    output_path = os.path.join('data', 'customer.csv')
    df.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")

except Exception as e:
    print(f"Error occurred: {str(e)}")