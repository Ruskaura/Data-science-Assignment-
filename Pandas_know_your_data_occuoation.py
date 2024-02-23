# import the necessary liberaries

import pandas as pd
# Import the dataset from this address.

import pandas as pd

# URL of the dataset
url = "https://drive.google.com/uc?id=1olOwT_0lQtaY7ZUFaLrFY-eu2P1DfT1y"

# Load the dataset into a dataframe
food = pd.read_csv(url)

# Display the first few rows of the dataframe to verify that it's loaded correctly
print(food.head())

# Assign it to a variable called users and use the 'user_id' as index
import pandas as pd

# URL of the dataset
url = "https://drive.google.com/uc?id=1olOwT_0lQtaY7ZUFaLrFY-eu2P1DfT1y"

# Load the dataset into a dataframe and set 'user_id' as index
users = pd.read_csv(url, index_col='user_id')

# Display the first few rows of the dataframe to verify that it's loaded correctly
print(users.head())

# See the first 25 entries
print(users.head(25))

# See the last 10 entries
print(users.tail(10))

# What is the number of observations in the dataset?

num_observations = users.shape[0]
print("Number of observations in the dataset:", num_observations)

# What is the number of columns in the dataset?

num_columns = users.shape[1]
print("Number of columns in the dataset:", num_columns)

# Print the name of all the columns.

column_names = users.columns.tolist()
print("Names of all the columns:")
for column in column_names:
    print(column)

# How is the dataset indexed?

index_type = users.index
print("Type of index in the dataset:", type(index_type))

# What is the data type of each column?

column_datatypes = users.dtypes
print("Data type of each column:")
print(column_datatypes)

# Print only the occupation column

occupation_datatype = users['occupation'].dtype
print("Data type of the 'occupation' column:", occupation_datatype)

#  How many different occupations are in this dataset?

num_occupations = len(users['occupation'].value_counts())
print("Number of different occupations:", num_occupations)

# What is the most frequent occupation?

most_frequent_occupation = users['occupation'].value_counts().idxmax()
print("Most frequent occupation:", most_frequent_occupation)

# Summarize the DataFrame.
summary = users.describe()
print(summary)

# Summarize all the columns
summary = users.describe(include='all')
print(summary)

# Summarize only the occupation column
occupation_summary = users['occupation'].describe()
print(occupation_summary)

# What is the mean age of users?

mean_age = users['age'].mean()
print("Mean age of users:", mean_age)

# What is the age with least occurrence?
least_occurrence_age = users['age'].value_counts().idxmin()
print("Age with least occurrence:", least_occurrence_age)
