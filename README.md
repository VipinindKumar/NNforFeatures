# NNforFeatures
Exloring automation of Feature Engineering using Neural Netwroks

<hr>

* Fixed seed for reproducible results (still will vary a little):
```python
seed = 43
import os
import random as rn
import numpy as np
import tensorflow as tf

os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
rn.seed(seed)
```


* Scale the data for 0 mean and unit std, using sklearn's StandardScaler():
```python
# adding scailing of the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scl_arr = scaler.fit_transform(X) #ndarray

X_scl = pd.DataFrame(X_scl_arr, columns=X.columns)
```


* Neural Network build using Keras library:
```python
input = Input(shape=(X_scl.shape[1],))

h1 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.03))
a1 = h1(input)

h2 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.03))
a2 = h2(a1)

h3 = Dense(4, activation='relu')
a3 = h3(a2)

output = Dense(1, activation='sigmoid')(a3)

model = Model(inputs=input, outputs=output)
```
