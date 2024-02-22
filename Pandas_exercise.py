import pandas as pd

# Load the dataset into a dataframe
food = pd.read_csv("https://drive.google.com/file/d/1olOwT_0lQtaY7ZUFaLrFY-eu2P1DfT1y/view?usp=drivesdk")

# Display the first few rows of the dataframe
print(food.head())

# see the first 5 entries 
print(food.head(5))
# what is the number of observations in the dataset

num_observations = food.shape[0]
print("Number of observations in the dataset:", [0,1,2,3,4,5,6,7,8,9])

#what is the number of column in the dataset
num_columns = food.shape[1]
print("Number of columns in the dataset:", [0,1,2,3,4,5,6,7,8,9])

#print the name of the column

column_names = food.columns.tolist()
print("Names of all the columns:")
for column in column_names:
    print(column)

# what is the name of column 105th 
column_105_name = food.columns[104]
print("Name of the 105th column:", column_105_name)

# What is the type of the observations of the 105th column?

column_105_type = food.iloc[:, 104].dtype
print("Type of observations in the 105th column:", column_105_type)

# How is the dataset indexed?

index_type = food.index
print("Type of index in the dataset:", type(index_type))

# What is the product name of the 19th observation?
product_name_19th_observation = food.loc[18, 'product_name']
print("Product name of the 19th observation:", product_name_19th_observation)

