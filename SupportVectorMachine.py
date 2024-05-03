import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import time

# Step 1: Read the original dataset
df = pd.read_csv('C:/Users/ABCD/Desktop/Computer Security/Datasets/transaction - used.csv')

# Step 2: Oversample the dataset to address class imbalance
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(df.drop('isFraud', axis=1), df['isFraud'])

# Step 3: Reduce the resampled dataset to 20k entries
df_sampled = pd.DataFrame(X_resampled, columns=df.drop('isFraud', axis=1).columns)
df_sampled['isFraud'] = y_resampled
df_sampled = df_sampled.sample(n=20000, random_state=42)

# Step 4: Plot a bar chart for the 'isFraud' column from the sampled dataset
plt.figure(figsize=(8, 6))
df_sampled['isFraud'].value_counts().plot(kind='bar')
plt.title('Distribution of isFraud (after sampling)')
plt.xlabel('isFraud')
plt.ylabel('Count')
plt.show()

# Step 5: Preprocess the data
X = df_sampled.drop('isFraud', axis=1)
y = df_sampled['isFraud']

# Encode categorical variables
# X = pd.get_dummies(X, columns=['Type'])
X['type'] = pd.factorize(X['type'])[0]
X['nameOrig'] = pd.factorize(X['nameOrig'])[0]
X['nameDest'] = pd.factorize(X['nameDest'])[0]

# Step 6: Split the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

# Step 7: Initialize the Support Vector Machine model
svm_model = SVC()

# Step 8: Train the model and measure the time taken
start_time = time.time()
svm_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Step 9: Predict the target variable for the training, validation, and testing sets
y_train_pred = svm_model.predict(X_train)
y_val_pred = svm_model.predict(X_val)
y_test_pred = svm_model.predict(X_test)

# Step 10: Calculate and print the accuracy for training, validation, and testing sets
train_accuracy = (y_train_pred == y_train).mean()
val_accuracy = (y_val_pred == y_val).mean()
test_accuracy = (y_test_pred == y_test).mean()

print("Training set accuracy:", train_accuracy)
print("Validation set accuracy:", val_accuracy)
print("Testing set accuracy:", test_accuracy)

# Step 11: Print the total time taken for training
print("Training time:", train_time)

# plot the result line chart

# Store the accuracy scores in lists
accuracy_scores = [train_accuracy, val_accuracy, test_accuracy]

labels = ['Training', 'Validation', 'Testing']
plt.plot(labels, accuracy_scores, marker='o')
plt.title('Accuracy Scores')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set the y-axis limits from 0 to 1
plt.grid(True)
plt.show()
