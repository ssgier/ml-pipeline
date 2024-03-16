from ml_pipeline.data import get_data
from ml_pipeline.dummy_model import DummyModel
from ml_pipeline.basic_nm_model import Config, BasicNMModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


data = get_data()


train_set = data.train_set
X_train_pre = train_set[0]
y_train_pre = train_set[1]

X_train, X_test, y_train, y_test = train_test_split(
    X_train_pre, y_train_pre, test_size=0.25, random_state=42
)

config = Config(
    N_in=784,
    N_out=10,
    ltp_step_up=0.003,
    ltp_step_down=0.0001,
    N_repeat=4,
    homeostasis_bump_factor=0.01,
    recent_rates_half_life=2000,
)


model = LogisticRegression(max_iter=1000, tol=1e-3)
model = BasicNMModel(config)
model.fit(X_train, y_train)

print(f"rates: {model._recent_rates.get_rates()}")
print(f"offsets: {model._homeostasis_offsets}")

print(f"score = {model.score(X_test, y_test)}")

# y_pred = model.predict(X_test)
# mask = y_pred != y_test

# offset = 1
# print(f"actual: {y_test[mask][offset]}, prediction: {y_pred[mask][offset]}")

# data = X_test[mask][offset].reshape((28, 28))
# plt.imshow(data, cmap="gray", interpolation="nearest")
# plt.show()

# exit(0)


w = [sw.reshape((28, 28)) for sw in model._weights]


fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns

# Flatten the axs array for easy iteration if it's multidimensional
axs = axs.flatten()

# Loop through the list of arrays and their corresponding axes to display each array
for i, ax in enumerate(axs):
    ax.imshow(w[i], cmap='gray')
    ax.set_title(i)  # Optional: add title to each subplot
    ax.axis("off")  # Optional: hide axes for cleaner visualization

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# mask = np.logical_and(y_pred == 9, y_test == 4)

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming you have a fitted model `model` and a test set (`X_test`, `y_test`)

# # Predict the test set
# y_pred = model.predict(X_test)

# # Generate the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Visualize the confusion matrix
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()
