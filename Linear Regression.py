import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Path to your dataset
file_path = 'C:/Users/HP/Desktop/plants.csv'  # Replace with your actual path

# Load dataset
df = pd.read_csv(file_path)

print("Initial Data:")
print(df.head())

# Fill missing values (if any)
df = df.fillna(method='ffill')

# Encode categorical variables
label_encoders = {}
for column in ['Plant Type', 'Soil Type', 'Time of Day', 'Plant Health', 'Growth Performance']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target
X = df.drop(columns=['Plant Needs'])
y = df['Plant Needs']

# Convert target variable to numerical values
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Ensure X is numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")

# Create DataFrame for comparison
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of Actual vs Predicted:")
print(comparison.head())

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', marker='o', linestyle='-')
plt.plot(y_pred, label='Predicted', marker='x', linestyle='--')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Function to predict new data
def predict_new_data(model, new_data, label_encoders):
    # Apply the same encoding as used during training
    for column, le in label_encoders.items():
        if column in new_data:
            new_data[column] = le.transform(new_data[column])
    new_data = pd.DataFrame([new_data])
    new_data = new_data.apply(pd.to_numeric, errors='coerce')
    
    # Predict using the model
    predictions = model.predict(new_data)
    return predictions

# Example new data for prediction
new_data = {
    'Plant Type': 'Sunflower',
    'Soil Type': 'Clay',
    'Sound Frequency (Hz)': 1500,
    'Time of Day': 'Morning',
    'Plant Health': 'Healthy',
    'Growth Performance': 'Good'
}

# Encode new data
for column, le in label_encoders.items():
    if column in new_data:
        new_data[column] = le.transform([new_data[column]])[0]

# Predict and print results
predicted_label = model.predict(pd.DataFrame([new_data]).apply(pd.to_numeric, errors='coerce'))
predicted_label = le_target.inverse_transform(predicted_label.astype(int))
print(f"\nPredicted Label for new data: {predicted_label[0]}")
