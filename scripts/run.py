from ml_pipeline.data import get_data
from ml_pipeline.dummy_model import DummyModel
from ml_pipeline.basic_nm_model import Config, BasicNMModel
from sklearn.model_selection import train_test_split


import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


data = get_data()

X_train, X_test, y_train, y_test = train_test_split(
    data.train_set[0], data.train_set[1], test_size=0.25, random_state=42
)

print(y_train[20:30])

config = Config(N_in=784, N_out=10, ltp_step_up=0.002, ltp_step_down=0.0001, N_repeat=1)


# model = SGDClassifier(max_iter=1000, tol=1e-3)
model = BasicNMModel(config)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Assuming you have a fitted model `model` and a test set (`X_test`, `y_test`)

# Predict the test set
y_pred = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
