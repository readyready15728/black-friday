import pandas as pd

df = pd.read_csv('black-friday.csv')

# Dropping User_ID and Product_ID as they are useless
#
# EDIT: Not useless
#
# df = df.drop(['User_ID', 'Product_ID'], axis=1)

# Product_Category_2 has many missing values (~31%) and appears bimodal so it
# was discarded; Product_Category_3 has a strong mode but the number of missing
# values is so massive (~69%) I did not want to carry out imputation
df = df.drop(['Product_Category_2', 'Product_Category_3'], axis=1)

df.to_csv('cleaned.csv', index=False)
