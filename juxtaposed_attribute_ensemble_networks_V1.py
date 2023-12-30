

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import zipfile
import io
import requests

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

##########
##########
# Select 70% of the data to generate the model and 30% to test it:

# Shuffle the DataFrame to randomize the rows
data_shuffled = data.sample(frac=1, random_state=42)  
# Calculate the split indices
split_index = int(0.7 * len(data_shuffled))  # 70% of the data
# Split the DataFrame into two portions (70% and 30%)
data_70_percent = data_shuffled.iloc[:split_index]
data_30_percent = data_shuffled.iloc[split_index:]
data = data_70_percent

# Divide the data_30_percent in two halves
split_index_30 = int(0.5 * len(data_30_percent))
data_15_percent_test_1 = data_30_percent.iloc[:split_index_30]
data_15_percent_test_2 = data_30_percent.iloc[split_index_30:]
##########
##########



#############################
X = data.drop(columns=[target_variable])
y = data[target_variable] 
       
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_normal = RandomForestRegressor(n_estimators=100, random_state=42)
rf_normal.fit(X_train, y_train)

#############################



# List of dictionaries
list_of_dictionaries = []

# List of dictionaries of R²
list_of_dictionaries_r_squared = []

# For each value of the target
for target_value in list(data[target_variable].unique()):

    # Generate auxiliary dataset
    dataset_aux = data[data[target_variable] == target_value]
    
    # Discard target in auxiliary dataset
    dataset_aux = dataset_aux.drop(columns=[target_variable])
    
    # Generate dictionary of ficticious targets and the models that predict them:
    dictionary_aux = {}
    # Correspondant dictionary of rmse for weighing 
    dictionary_aux_r_squared = {}
    
    for fict_target in dataset_aux.columns.tolist():
        
        
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
            # (*)Actually, negative R² values will be converted to 0,
            # so it is not really R²
            predictions = rf.predict(X_test)
            y_mean = np.mean(y_test)
            # Calculate the total sum of squares
            tss = np.sum((y_test - y_mean) ** 2)
            # Calculate the residual sum of squares
            rss = np.sum((y_test - predictions) ** 2)
            # Calculate R² score
            r_squared = 1 - (rss / tss)    
            # Apply modification
            if (r_squared < 0):
                r_squared = 0
            print(r_squared)    
            
            dictionary_aux_r_squared[fict_target] = r_squared
        
    list_of_dictionaries.append(dictionary_aux)    
    list_of_dictionaries_r_squared.append(dictionary_aux_r_squared)    

list_unique_values_target = list(data[target_variable].unique())
    
# data_15_percent_test_1 = data_15_percent_test_1.head(100)
target_test_values_real = data_15_percent_test_1[target_variable].tolist()
target_test_values_predicted = []
# For each register in data_15_percent_test_1 dataset
number_of_rows = len(data_15_percent_test_1)
for i in range(0, number_of_rows):
    row = data_15_percent_test_1.iloc[i]
    # Convert the row to a dataframe object of one row
    row = row.to_frame().T
    
    # List of summatories of rmse
    list_sum_rmse = []
    
    # For each value of the target
    for case in range(0, len(list_unique_values_target)):

        dictionary_case = list_of_dictionaries[case]
        dictionary_case_r_squared = list_of_dictionaries_r_squared[case]

        sum_rmse = 0    
        
        for fict_target in dictionary_case:
            X = row.drop(columns=[target_variable, fict_target])
            y_predicted = dictionary_case[fict_target].predict(X)
            y_real = row[fict_target]         
            mse = mean_squared_error(y_real, y_predicted)

            rmse = np.sqrt(mse)
            # Weigh according to r squared of the model (THE WEIGHING IMPROVEMENT
            # WILL BE IMPLEMENTED IN FUTURE VERSION)
            # rmse = rmse * dictionary_case_r_squared[fict_target]
            
            #Add to summatory
            sum_rmse = sum_rmse + rmse
   
        list_sum_rmse.append(sum_rmse)     
    
    
    min_index = list_sum_rmse.index(min(list_sum_rmse))
    
    target_predicted = list_unique_values_target[min_index]
    target_test_values_predicted.append(target_predicted)
    prob = 1 - (list_sum_rmse[min_index] / sum(list_sum_rmse))

    

X = data_15_percent_test_1.drop(columns=[target_variable])
target_test_values_predicted_normal = rf_normal.predict(X)
target_test_values_real_normal = data_15_percent_test_1[target_variable]         
precision_normal = precision_score(target_test_values_real_normal, target_test_values_predicted_normal)
recall_normal = recall_score(target_test_values_real_normal, target_test_values_predicted_normal)
f1_normal = f1_score(target_test_values_real_normal, target_test_values_predicted_normal)

print("Normal Precision:", precision_normal)
print("Normal Recall (Sensitivity):", recall_normal)
print("Normal F1 Score:", f1_normal)


precision = precision_score(target_test_values_real, target_test_values_predicted)
recall = recall_score(target_test_values_real, target_test_values_predicted)
f1 = f1_score(target_test_values_real, target_test_values_predicted)

print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("F1 Score:", f1)






