from ml_pipeline.data import get_data
from ml_pipeline.basic_nm_model import Config, BasicNMModel
from sklearn.model_selection import train_test_split
import json
import sys


params = json.loads(sys.argv[1])
seed = int(sys.argv[2])
config = Config(**params, N_in=784, N_out=10, N_repeat=4)

data = get_data()

X_train, X_test, y_train, y_test = train_test_split(
    data.train_set[0], data.train_set[1], test_size=0.25, random_state=seed
)

model = BasicNMModel(config)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

obj_func_val = 1 - score

result = {"objFuncVal": obj_func_val}

print(json.dumps(result))
