import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 1. Load and Preprocess your local dataset
data = pd.read_csv("churn-bigml-80.csv")
data = data.drop_duplicates().reset_index(drop=True)

# Mappings and Encoding
data["International plan"] = data["International plan"].map({'Yes':1,'No':0})
data["Voice mail plan"] = data["Voice mail plan"].map({'Yes':1,'No':0})

encoder = OneHotEncoder(sparse_output=False)
encodedData = pd.DataFrame(encoder.fit_transform(data[["State"]]), columns=encoder.get_feature_names_out(["State"]))
data = pd.concat([data, encodedData], axis=1).drop(columns=['State'])

# Robust Scaling
numericalData = ["Account length","Number vmail messages","Total day minutes","Total day calls","Total day charge","Total eve minutes","Total eve calls","Total eve charge","Total night minutes","Total night calls","Total night charge","Total intl minutes","Total intl calls","Total intl charge","Customer service calls"]
data[numericalData] = RobustScaler().fit_transform(data[numericalData])

# X and y
X = data.drop(columns=["Churn", "Area code"])
y = data["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Initialize the SVM models (Linear vs RBF)
svm_linear = SVC(kernel='linear', probability=True, random_state=42)
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)

# 3. Train and Compare
for name, model in [("SVM with Linear Kernel", svm_linear), ("SVM with RBF Kernel", svm_rbf)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"AUC:       {roc_auc_score(y_test, y_prob):.4f}\n")
    
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

print("Generating 2D Decision Boundary Plot via PCA...")

# 1. Squash the 70+ feature dataset down to exactly 2 dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 2. A 2D plot requires a model trained on exactly 2 features. 
# We will train a quick visualization-only RBF model on the PCA data.
svm_viz = SVC(kernel='rbf', random_state=42)
svm_viz.fit(X_pca, y)

# 3. Create the Plot
fig, ax = plt.subplots(figsize=(10, 7))

# Draw the background decision boundary zones (Red vs Blue)
DecisionBoundaryDisplay.from_estimator(
    svm_viz,
    X_pca,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.6,
    ax=ax,
    eps=0.5
)

# Plot the actual customer data points on top
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=25)

#labels and legend
plt.title("SVM Decision Boundary (RBF Kernel) via PCA")
plt.xlabel("Principal Component 1 (Squashed Data)")
plt.ylabel("Principal Component 2 (Squashed Data)")

# Create a legend to show which color is which
handles, _ = scatter.legend_elements()
plt.legend(handles, ["Stayed (0)", "Churned (1)"], loc="best")

plt.show()