#ANOMALY DETECTION OF FINANCIAL TRANSACTION USING ISOLATION FOREST
# Set the random seed for reproducibility:
np.random.seed(42)
# Generate the normal financial transactions:
normal_transactions = pd.DataFrame({
    'Amount': np.random.normal(loc=100, scale=20, size=1000),
    'Merchant': np.random.choice(['A', 'B', 'C', 'D'], size=1000),
    'Category': np.random.choice(['Retail', 'Food', 'Travel'], size=1000),
})
# Generate the anomalous transactions:
anomalous_transactions = pd.DataFrame({
    'Amount': np.random.normal(loc=1000, scale=200, size=50),
    'Merchant': np.random.choice(['E', 'F'], size=50),
    'Category': np.random.choice(['Other'], size=50),
})
# Combine the normal and anomalous transactions into a single dataset:
dataset = pd.concat([normal_transactions, anomalous_transactions], ignore_index=True)
# Shuffle the dataset:
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
# Assuming 'dataset' contains the finance-related dataset created

# Encode categorical columns using one-hot encoding
dataset_encoded = pd.get_dummies(dataset, columns=['Merchant', 'Category'])

# Create the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)

# Fit the model on the encoded dataset
model.fit(dataset_encoded)

# Predict the anomalies (1 for normal, -1 for anomaly)
predictions = model.predict(dataset_encoded)

# Retrieve the anomaly scores
anomaly_scores = model.decision_function(dataset_encoded)

# Combine the predictions and anomaly scores with the original dataset
results = pd.DataFrame({'Transaction': dataset.index, 'Prediction': predictions, 'Anomaly Score': anomaly_scores})
results['Prediction'] = np.where(results['Prediction'] == 1, 'Normal', 'Anomaly')

# Print the results
print(results)



