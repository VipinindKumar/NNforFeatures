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



* model.summary():
```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 8)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                288       
_________________________________________________________________
dense_2 (Dense)              (None, 32)                1056      
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 132       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 5         
=================================================================
Total params: 1,481
Trainable params: 1,481
Non-trainable params: 0
_________________________________________________________________
```



* Adam Optimizer:
```python
adam_opt = Adam(learning_rate=0.001,beta_1=0.9, beta_2=0.999, amsgrad=False)
```



* Loss and Metric:
```python
model.compile(loss='mean_squared_error', optimizer=adam_opt, metrics=['accuracy'])
```



* .fit() Hypreparameters:
```python
fit = model.fit(X_scl, Y, epochs=500, validation_split=0.3)
```
