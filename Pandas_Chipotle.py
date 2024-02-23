# Import the necessary libraries
import pandas as pd
# Import the dataset from this address.

import pandas as pd

# Replace 'your_dataset_url' with the actual URL of your dataset
dataset_url = 'https://drive.google.com/uc?id=1olOwT_0lQtaY7ZUFaLrFY-eu2P1DfT1y"https://drive.google.com/uc?id=1olOwT_0lQtaY7ZUFaLrFY-eu2P1DfT1y'
df = pd.read_csv(dataset_url)

# Assign it to a variable called chipo.
import pandas as pd

# Replace 'your_dataset_url' with the actual URL of your dataset
dataset_url = 'https://drive.google.com/uc?id=1olOwT_0lQtaY7ZUFaLrFY-eu2P1DfT1y"https://drive.google.com/uc?id=1olOwT_0lQtaY7ZUFaLrFY-eu2P1DfT1y'
chipo = pd.read_csv(dataset_url)

# See the first 10 entries
print(chipo.head(10))

# What is the number of observations in the dataset?
#solution 1 
num_observations = chipo.shape[0]
print("Number of observations in the dataset:", num_observations)
# solution 2 
num_observations = len(chipo)
print("Number of observations in the dataset:", num_observations)

# What is the number of columns in the dataset?
num_columns = chipo.shape[1]
print("Number of columns in the dataset:", num_columns)

# Print the name of all the columns.
column_names = chipo.columns.tolist()
print("Names of all the columns:")
for column in column_names:
    print(column)

# How is the dataset indexed?
index_type = chipo.index
print("Type of index in the dataset:", type(index_type))

# Which was the most-ordered item?
most_ordered_item = chipo.groupby('item_name')['quantity'].sum().idxmax()
print("Most-ordered item:", most_ordered_item)

# For the most-ordered item, how many items were ordered?
most_ordered_quantity = chipo.groupby('item_name')['quantity'].sum().max()
print("Number of items ordered for the most-ordered item:", most_ordered_quantity)

# What was the most ordered item in the choice_description column?
most_ordered_choice = chipo.groupby('choice_description')['quantity'].sum().idxmax()
print("Most ordered item in the choice_description column:", most_ordered_choice)

# How many items were orderd in total?
total_items_ordered = chipo['quantity'].sum()
print("Total items ordered:", total_items_ordered)

# Turn the item price into a float
# Remove the dollar sign and convert to float
chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]))

# Verify the changes
print(chipo.head())

# Check the item price type
item_price_type = chipo['item_price'].dtype
print("Data type of the 'item_price' column:", item_price_type)

# Create a lambda function and change the type of item price
# Define a lambda function to remove the dollar sign and convert to float
remove_dollar_and_convert_to_float = lambda x: float(x[1:])

# Apply the lambda function to the 'item_price' column
chipo['item_price'] = chipo['item_price'].apply(remove_dollar_and_convert_to_float)

# Verify the changes
print(chipo.head())

# Check the item price type

item_price_type = chipo['item_price'].dtype
print("Data type of the 'item_price' column:", item_price_type)

# How much was the revenue for the period in the dataset?
# Calculate revenue for each order
chipo['revenue'] = chipo['quantity'] * chipo['item_price']

# Sum up the total revenue
total_revenue = chipo['revenue'].sum()
print("Total revenue for the period:", total_revenue)
# How many orders were made in the period?
num_orders = chipo['order_id'].nunique()
print("Number of orders made in the period:", num_orders)

# What is the average revenue amount per order?
# Solution 1
average_revenue_per_order = total_revenue / num_orders
print("Average revenue amount per order (Solution 1):", average_revenue_per_order)

# Solution 2
average_revenue_per_order = chipo['revenue'].mean()
print("Average revenue amount per order (Solution 2):", average_revenue_per_order)

# How many different items are sold?
num_unique_items = chipo['item_name'].nunique()
print("Number of different items sold:", num_unique_items)

