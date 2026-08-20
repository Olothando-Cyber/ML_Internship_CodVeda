import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
data = pd.read_csv("churn-bigml-80.csv")
data = data.drop_duplicates().reset_index(drop = True)

# Mappings and Encoding
data["International plan"] = data["International plan"].map({'Yes':1,'No':0})
data["Voice mail plan"] = data["Voice mail plan"].map({'Yes':1,'No':0})
encoder = OneHotEncoder(sparse_output=False)
encodedData = pd.DataFrame(encoder.fit_transform(data[["State"]]), columns=encoder.get_feature_names_out(["State"]))
data = pd.concat([data, encodedData], axis=1).drop(columns=['State'])
# Robust Scaling
numericalData = ["Account length","Number vmail messages","Total day minutes","Total day calls","Total day charge","Total eve minutes","Total eve calls","Total eve charge","Total night minutes","Total night calls","Total night charge","Total intl minutes","Total intl calls","Total intl charge","Customer service calls"]
data[numericalData] = RobustScaler().fit_transform(data[numericalData])
X = data.drop(columns=["Churn", "Area code"])
y = data["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = Sequential()
model.add(Dense(64,activation='relu',input_shape=(X_train.shape[1],)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Training Neural Network...")
# epochs=50 means the network will look at the entire dataset 50 times to learn
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate on the unseen test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# Plot the Learning Curve (Loss over time)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Learning Curve')
plt.xlabel('Epoch (Training Cycles)')
plt.ylabel('Error (Loss)')
plt.legend()
plt.grid(True)
plt.show()