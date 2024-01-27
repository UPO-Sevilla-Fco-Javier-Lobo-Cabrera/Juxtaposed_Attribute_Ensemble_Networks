

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import zipfile
import io
import requests
import time

###########################
# Download dataset from UCI* for the first time and save it locally:
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
# response = requests.get(url)
# z = zipfile.ZipFile(io.BytesIO(response.content))
# Specify the name of the CSV file to read from the ZIP file
# csv_filename = 'bank-additional/bank-additional.csv'
# Read the specified CSV file into a DataFrame
# with z.open(csv_filename) as file:
#    data = pd.read_csv(file, sep=';')
# print(data.columns)
# Save the dataset to a CSV file locally
# data.to_csv('dataset.csv', index=False)

#(*)Citation:  
# Moro,S., Rita,P., and Cortez,P.. (2012). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
###########################

###########################

# Reading the dataset from the saved file
data = pd.read_csv('dataset.csv')
# data = data.sample(frac=0.2, random_state=42)  
###########################

data = data.drop_duplicates()

print(data['y'].value_counts())

print(data.shape)

# Handling missing values (drop rows with missing values for simplicity)
data.dropna(inplace=True)

print(data.shape)

# Encoding categorical variables using one-hot encoding
data = pd.get_dummies(data)

# Name of the target of the dataset (after one-hot encoding)
target_variable = 'y_yes'

# Normalizing variables:
data_columns = data.columns
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns=data_columns)


# Denormalize target values (these must be 0 or 1):
def aux_denormalize_target(aux):
    threshold = min(list(data[target_variable].unique()))
    if aux > threshold:
        return 1
    else:
        return 0

data[target_variable] = data[target_variable].apply(aux_denormalize_target)

# Drop the other column obtained from one-hot encoding the target:
data = data.drop(columns=["y_no"])


print(data.columns)

##########
##########

# Shuffle the DataFrame to randomize the rows
data_shuffled = data.sample(frac=1, random_state=42)  
# Calculate the split indices
split_index = int(0.6 * len(data_shuffled))  # 60% of the data
# Split the DataFrame into two portions (60% and 40%)
data_60_percent = data_shuffled.iloc[:split_index]
data_40_percent = data_shuffled.iloc[split_index:]
data = data_40_percent

# Divide the data_60_percent into a group of 50% and another of 10%
split_index = int(50/(50+10) * len(data_60_percent))
data_2 = data_60_percent.iloc[:split_index]
data_test = data_60_percent.iloc[split_index:]
##########
##########



#############################
X = data.drop(columns=[target_variable])
y = data[target_variable] 
       
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_normal = RandomForestClassifier(n_estimators=100, random_state=42)
rf_normal.fit(X_train, y_train)


#############################



# List of dictionaries
list_of_dictionaries = []

# List of dictionaries of R²
list_of_dictionaries_r_squared = []

# For each value of the target
for target_value in sorted(list(data[target_variable].unique())):
    print(target_value)
    print("#########################################")

    # Generate auxiliary dataset
    dataset_aux = data[data[target_variable] == target_value]
    
    # Discard target in auxiliary dataset
    dataset_aux = dataset_aux.drop(columns=[target_variable])
    
    # Generate dictionary of ficticious targets and the models that predict them:
    dictionary_aux = {}
    # Correspondant dictionary of rmse for weighing 
    dictionary_aux_r_squared = {}
    
    for fict_target in dataset_aux.columns.tolist():
        print(fict_target)
        
        
        # Train the Random Forest model and save it
        X = dataset_aux.drop(columns=[fict_target])
        y = dataset_aux[fict_target] 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
        # Fit a regressor:
        if True:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            dictionary_aux[fict_target] = rf
            
            #####
            # Computation of R²* for weighing:
            # (*)Actually, it is a variation of R² so that the values are
            # in the range [0, 1] negative R² values will be converted to 0,
            # so it is not really R²
            predictions = rf.predict(X_test)
            y_mean = np.mean(y_test)
            # Calculate the total sum of squares
            tss = np.sum((y_test - y_mean) ** 2)
            # Calculate the residual sum of squares
            rss = np.sum((y_test - predictions) ** 2)
            # Calculate R² score
            # If tss == 0 then R² will be 1
            if (tss < 0.00001) & (tss > -0.00001):
                r_squared = 1
            else:    
                r_squared = 1 - (rss / tss)    
            # Apply modification
            # if (r_squared < 0):
            #    r_squared = 0
            # print(r_squared)    
            
            dictionary_aux_r_squared[fict_target] = r_squared
        
    list_of_dictionaries.append(dictionary_aux)    
    list_of_dictionaries_r_squared.append(dictionary_aux_r_squared)    

list_unique_values_target = sorted(list(data[target_variable].unique()))
    
target_test_values_real = data_2[target_variable].tolist()
data_2.to_csv("dataframe_or_info.csv")

list_of_rows_dataframe_new = []


# For each register in data_2 dataset
number_of_rows = len(data_2)
print(number_of_rows)
time.sleep(5)

for i in range(0, number_of_rows):
    print(i)
    row = data_2.iloc[i]
    # Convert the row to a dataframe object of one row
    row = row.to_frame().T
    
    # List of sublists of rmse
    list_rmse = []
    
    # For each value of the target
    for case in range(0, len(list_unique_values_target)):

        dictionary_case = list_of_dictionaries[case]
        dictionary_case_r_squared = list_of_dictionaries_r_squared[case]

        sub_list_rmse = []
        
        for fict_target in dictionary_case:
            X = row.drop(columns=[target_variable, fict_target])
            y_predicted = dictionary_case[fict_target].predict(X)
            y_real = row[fict_target]         
            mse = (y_real - y_predicted) ** 2

            rmse = np.sqrt(mse)
            rmse = float(round(rmse.iloc[0], 4))
            # Weigh according to r squared of the model 
            rmse = rmse * (dictionary_case_r_squared[fict_target])**2
            # rmse = rmse * ((list_of_dictionaries_r_squared[0][fict_target] + list_of_dictionaries_r_squared[1][fict_target])/2)**2
            
            # Weigh also according to the difference of y_predicted between
            # the two models. Multiply by 10 to make the difference > 1 in
            # most of the cases. This promotes high weight if difference > 0.1
            # but low weight if difference < 0.1 
            # rmse = rmse * abs(list_of_dictionaries[0][fict_target].predict(X) - list_of_dictionaries[1][fict_target].predict(X))
            # rmse = rmse * 10 * abs(list_of_dictionaries[0][fict_target].predict(X) - list_of_dictionaries[1][fict_target].predict(X))
            
            # Add to sublist
            sub_list_rmse.append(rmse)
   
        list_rmse.append(sub_list_rmse)     
    
    
    
    row_to_append = []
    for h in range(0, len(sub_list_rmse)):
        row_to_append.append(list_rmse[0][h])
        row_to_append.append(list_rmse[1][h])
    # Include the target column in the row    
    row_to_append.append(row.iloc[0][target_variable])    
    list_of_rows_dataframe_new.append(row_to_append)

names_cols_dataframe_new = []
for u in dictionary_case.keys():
    if (u != target_variable):
        names_cols_dataframe_new.append(u + "_0")
        names_cols_dataframe_new.append(u + "_1")        
names_cols_dataframe_new.append(target_variable)
dataframe_new = pd.DataFrame(list_of_rows_dataframe_new, columns=names_cols_dataframe_new)
dataframe_new.to_csv('dataframe_new.csv', index=False)




# df = pd.read_csv("dataframe_new.csv")  
df = dataframe_new
df = df.drop(columns=[target_variable])

cols_a = data_2.columns.to_list()
cols_b = df.columns.to_list()

data_2 = data_2.reset_index(drop=True)
df = df.reset_index(drop=True)


# Concatenate horizontally
result_df = pd.concat([df, data_2], axis=1, ignore_index=True)
result_df.columns = cols_b + cols_a

# The process has generated additional columns in the dataframe (those ending with _0 or _1).
# These additional columns could enhance potentially performance.
# The whole cycle may be repeated again (sort of a new layer) generating more additional
# variables (these will contain also those now ending with _0_0, _0_1, _1_0, and 1_1).


features = result_df.drop(target_variable, axis=1)
target = result_df[target_variable]

print(target.value_counts())


# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform k-fold cross-validation (k=5)
cv_predictions = cross_val_predict(rf_model, features, target, cv=5)

# Calculate precision, recall, and F1 scores
precision = precision_score(target, cv_predictions)
recall = recall_score(target, cv_predictions)
f1 = f1_score(target, cv_predictions)

# Print the precision, recall, and F1 scores
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')




