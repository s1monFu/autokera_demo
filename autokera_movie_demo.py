import pandas as pd
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import tensorflow as tf
import keras.backend as K

# Clear previous sessions to free up resources
K.clear_session()

# Step 1: Read the dataset
df = pd.read_csv('./movies.csv')

# Step 2: Process the genres
df['genres_list'] = df['genres'].str.split('|')

# Extract the primary genre
df['primary_genre'] = df['genres_list'].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
)

# Remove rows with missing primary_genre or overview
df = df[df['overview'].notnull() & df['primary_genre'].notnull()]

# Ensure that overview is a string
df['overview'] = df['overview'].astype(str)

# Step 3: Prepare features and labels
x = df['overview'].values
y = df['primary_genre'].values

# Combine x and y into a DataFrame for easier manipulation
data = pd.DataFrame({'overview': x, 'primary_genre': y})

data = data.iloc[:100]

# **New Step**: Remove classes with fewer than 2 samples
class_counts = data['primary_genre'].value_counts()
classes_with_enough_samples = class_counts[class_counts >= 2].index.tolist()
data = data[data['primary_genre'].isin(classes_with_enough_samples)]

# Shuffle the data to ensure randomness
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Limit to 500 samples to manage memory usage


# Extract x and y from the DataFrame
x = data['overview'].values
y = data['primary_genre'].values

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Verify class counts after filtering
class_counts_after = Counter(y_encoded)
print("Class counts after filtering:", class_counts_after)

# Convert to proper types
x = np.array(x, dtype=str)
y_encoded = np.array(y_encoded, dtype=int)

# Step 4: Split into training and testing sets with stratification
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Step 5: Initialize the text classifier
clf = ak.TextClassifier(overwrite=True, max_trials=1)

# Train the classifier
clf.fit(
    x_train,
    y_train,
    validation_split=0.15,
    epochs=1,
    batch_size=2,
)

# Step 6: Evaluate the classifier
predicted_y = clf.predict(x_test)
evaluation = clf.evaluate(x_test, y_test)
print(f"Evaluation Results: {evaluation}")

# Optional: Decode the predicted labels back to genres
predicted_genres = le.inverse_transform(predicted_y.flatten().astype(int))
print(f"Predicted Genres: {predicted_genres}")
