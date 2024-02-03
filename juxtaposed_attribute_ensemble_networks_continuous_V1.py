

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
print("Obtaining dataset...")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
# Specify the name of the CSV file to read from the ZIP file
csv_filename = 'bank-additional/bank-additional.csv'
# Read the specified CSV file into a DataFrame
with z.open(csv_filename) as file:
    data = pd.read_csv(file, sep=';')

#(*)Moro,S., Rita,P., and Cortez,P.. (2012). Bank Marketing. UCI Machine
# Learning Repository. https://doi.org/10.24432/C5K306.
###########################


# Sampling of the data
data = data.sample(frac=0.6, random_state=42)  


data = data.drop_duplicates()

# Handling missing values (drop rows with missing values for simplicity)
data.dropna(inplace=True)


# Encoding categorical variables using one-hot encoding
data = pd.get_dummies(data)

# Name of the target of the dataset (after one-hot encoding)
target_variable = 'age'

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


# List of dictionaries
list_of_dictionaries = []

# List of dictionaries of R²
list_of_dictionaries_r_squared = []

if True:
    # Generate auxiliary dataset    
    dataset_aux = data.drop(columns=[target_variable])
    
    # Generate dictionary of ficticious targets and the models that predict them:
    dictionary_aux = {}
    # Correspondant dictionary of rmse for weighing 
    dictionary_aux_r_squared = {}
    
    # Creation of model for each ficticious target (i.e attribute variable)
    for fict_target in dataset_aux.columns.tolist():
        print("Creating model for " + str(fict_target) + "...")
        
        
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
            # Computation of R² for weighing:
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
            
            dictionary_aux_r_squared[fict_target] = r_squared
        
    list_of_dictionaries.append(dictionary_aux)    
    list_of_dictionaries_r_squared.append(dictionary_aux_r_squared)    


list_unique_values_target = sorted(list(data[target_variable].unique()))
    
list_of_rows_dataframe_new = []

# For each register calculate new columns
number_of_rows = len(data)
print("Number of rows to process: " + str(number_of_rows))
time.sleep(5)

for i in range(0, number_of_rows):
    print(i)
    row = data.iloc[i]
    # Convert the row to a dataframe object of one row
    row = row.to_frame().T
    
    # List of sublists of rmse (this is auxiliary for later creating the new
    # columns)
    list_rmse = []
    
    if True:

        # Obtain dictionary of attribute models and dictionary of correspondant
        # R² values of those models
        dictionary_case = list_of_dictionaries[0]
        dictionary_case_r_squared = list_of_dictionaries_r_squared[0]

        sub_list_rmse = []
        
        for fict_target in dictionary_case:
            # Calculation of the predicted value of the attribute
            X = row.drop(columns=[target_variable, fict_target])
            y_predicted = dictionary_case[fict_target].predict(X)
            # Calculation of difference between predicted value of the attribute 
            # and actual value. The difference is calculated in form of rmse
            # and then weighed according to the R² of the predictive model for
            # the attribute in question
            y_real = row[fict_target]         
            mse = (y_real - y_predicted) ** 2
            rmse = np.sqrt(mse)
            rmse = float(round(rmse.iloc[0], 4))
            # Weigh according to r squared of the model 
            rmse = rmse * (dictionary_case_r_squared[fict_target])**2
            
            # Add to sublist
            sub_list_rmse.append(rmse)
   
        list_rmse.append(sub_list_rmse)     
    
    
    # Generate row of dataframe containing only the new columns
    # (this dataframe will be later concatenated horizontally
    # to the original dataframe, so that the resulting dataframe
    # contains both the original and the new columns):
    row_to_append = []
    # Add the columns that contain the difference between the
    # predicted attribute value and the actual one)
    for h in range(0, len(sub_list_rmse)):
        row_to_append.append(list_rmse[0][h])
        
    list_of_rows_dataframe_new.append(row_to_append)

# Generate dataframe that contains only the new columns
names_cols_dataframe_new = []
for u in dictionary_case.keys():
    if (u != target_variable):
        names_cols_dataframe_new.append(u + "_new")
                
dataframe_new = pd.DataFrame(list_of_rows_dataframe_new, columns=names_cols_dataframe_new)

###################################################################
# Concatenation of new columns to the original dataframe

df = dataframe_new

cols_a = data.columns.to_list()
cols_b = df.columns.to_list()

data = data.reset_index(drop=True)
df = df.reset_index(drop=True)

# Concatenate horizontally to add new columns
result_df = pd.concat([df, data], axis=1, ignore_index=True)
result_df.columns = cols_b + cols_a


# The process has generated additional columns in the dataframe (those ending with _0 or _1).
# These additional columns could enhance potentially performance.
# The whole cycle may be repeated again (sort of a new layer) generating more additional
# variables (these will contain also those now ending with _0_0, _0_1, _1_0, and 1_1).

###################################################################

# Comparison between performance metrics of the updated dataset and
# the normal (original) dataset. For this purpose, an RF model using 
# cross validation throughout all rows is employed in both cases
# with default sklearn RF hyperparameters, which imply max_depth=None
# so that the trees can potentially grow very deep.

# Obtain metrics of the updated dataset 
features = result_df.drop(target_variable, axis=1)
target = result_df[target_variable]
# Initialize RandomForestClassifier
rf_model = RandomForestRegressor(random_state=42)
# Perform k-fold cross-validation (k=5)
cv_predictions = cross_val_predict(rf_model, features, target, cv=5)
# Calculate rmse
rmse = np.sqrt(mean_squared_error(target, cv_predictions))
print("RMSE:", rmse)


# Obtain metrics of the normal (i.e original) dataset
features = data.drop(target_variable, axis=1)
target = data[target_variable]
rf_normal = RandomForestRegressor(random_state=42)
# Perform k-fold cross-validation (k=5)
cv_predictions = cross_val_predict(rf_normal, features, target, cv=5)
rmse = np.sqrt(mean_squared_error(target, cv_predictions))
print("Normal RMSE:", rmse)


