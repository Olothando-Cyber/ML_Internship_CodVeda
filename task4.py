import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score

# 1. Load your local dataset
data = pd.read_csv("1) iris.csv")
print(data.duplicated().count())
data = data.drop_duplicates().reset_index(drop=True)

# 2. Set X and y (No scaling needed for Decision Trees!)
X = data.drop(columns=['species'])
y = data['species']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train and Prune the Decision Tree (max_depth=3 prevents overfitting)
clf_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_pruned.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = clf_pruned.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Model F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# 5. Visualize the tree structure
plt.figure(figsize=(12, 8))
plot_tree(clf_pruned, 
          feature_names=X.columns,  
          class_names=clf_pruned.classes_,
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Pruned Decision Tree for Iris Classification")
plt.show()