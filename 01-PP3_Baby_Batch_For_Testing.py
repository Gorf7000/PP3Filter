#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Define file paths
input_file = 'data/batch_for_analysis - Original.csv'
output_file = 'data/batch_for_analysis.csv'

# Load the CSV file into a DataFrame, drop "Unnamed: 0.1", rename "Unnamed: 0" to "Index"
df = pd.read_csv(input_file).drop(columns=['Unnamed: 0.1']).rename(columns={'Unnamed: 0': 'Index'})

# Get user input for sample size
sample_size = input("Enter the sample size (press Enter to use the entire dataset): ")
if sample_size:
    sample_size = int(sample_size)
else:
    sample_size = len(df)

# Get user input for using a random seed
use_seed = input("Do you want to use a random seed? (Y/N): ").strip().upper()

if use_seed == 'Y':
    seed_input = input("Enter the random seed (press Enter to use the default seed 4): ")
    if seed_input:
        seed = int(seed_input)
    else:
        seed = 4
    df_sample = df.sample(n=sample_size, random_state=seed)
else:
    df_sample = df.sample(n=sample_size)

# Save the sampled DataFrame back to the original file
df_sample.to_csv(output_file, index=False)

print(f"Sampled dataset with {sample_size} records saved to {output_file}")


# In[ ]:




