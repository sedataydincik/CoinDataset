import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Dataset loading from GitHub
url = "https://github.com/sedataydincik/CoinDataset/raw/main/CoinDataset.csv"
df_train = pd.read_csv(url)

# Converting labels
label_encoder = LabelEncoder()
df_train['Landed Coin'] = label_encoder.fit_transform(df_train['Landed Coin'])
df_train['Drop Orientation'] = label_encoder.fit_transform(df_train['Drop Orientation'])

# Label defining for training
X = df_train[['Drop Orientation']]
y = df_train['Landed Coin']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Selected Model for training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction for test and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# Coin orientation info from user
user_orientation = input("Please enter the coin drop orientation (Heads Facing Up, Tails Facing Up, Vertical): ")
user_orientation_encoded = label_encoder.transform([user_orientation])[0]

# Prediction for input
user_input = pd.DataFrame([[user_orientation_encoded]], columns=['Drop Orientation'])
predicted_probability = model.predict_proba(user_input)[0]

# Percentage of probabilities
print(f"Heads probability: {predicted_probability[0] * 100:.2f}%")
print(f"Tails probability: {predicted_probability[1] * 100:.2f}%")

# Calculate and print the average Distance to Origin for the selected orientation
selected_orientation_avg_distance = df_train[df_train['Drop Orientation'] == user_orientation_encoded]['Distance to origin'].mean()
print(f"Average Distance to origin for the selected {user_orientation} orientation: {selected_orientation_avg_distance}")
