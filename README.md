# NNforFeatures
Exloring automation of Feature Engineering using Neural Netwroks

<hr>

- Need to do:
    - New Dataset (bigger and with diverse uncorrelated features)
    - ~~abstract function to utilize both tanh and leakyrelu as activation~~


<hr>


* To start Pima Indians Diabetes Database from Kaggle is used:
```
Columns = Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
shape = (768, 8)
```



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


* Scale the data for 0 mean and unit std, using sklearn's StandardScaler()


* Neural Network build using Keras library, model.summary():
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
<hr>



* XGBoost model to measure the features importance:
```python
model = XGBClassifier(max_depth=7,
                      random_state=seed,
                      min_child_weight=15, 
                      learning_rate=0.001,
                      n_estimators=400)
```



* Feature Importamce in XGBoost model:

![pima dataset feature importance](images/pima_feature_importance.png)



* Getting Weights and bias for hidden layer and then Calculating activation for hidden layers:
```python
# calculate activation for hidden layers
# a = g(z) = g(wx + b)
# (nl,m)   (nl,n)(n,m)  (nl,1)
```
<hr>

* a1 mostly 0.0000s, sparse dataframe::
    - more hidden units than required?
    - check % of zeroes in features
    - L1 regularization?
- a1.sum():
    - only 6 non-zero features
```
f1-0     129.157055
f1-1       0.000000
f1-2       0.000000
f1-3       0.000000
f1-4       0.000000
f1-5       0.000000
f1-6       0.000000
f1-7       0.000000
f1-8       0.000000
f1-9       0.000000
f1-10    192.624494
f1-11      0.000000
f1-12      0.000000
f1-13      3.890168
f1-14      0.000000
f1-15    119.354960
f1-16      0.000000
f1-17    144.222696
f1-18      0.000000
f1-19      0.000000
f1-20      0.000000
f1-21      0.000000
f1-22      0.000000
f1-23      0.000000
f1-24      0.000000
f1-25      0.000000
f1-26      0.000000
f1-27    120.874367
f1-28      0.000000
f1-29      0.000000
f1-30      0.000000
f1-31      0.000000
dtype: float64
```

- RELU activation function induuce sparsity

```
From: Xavier Glorot, Antoine Bordes and Yoshua Bengio
```

![](images/reluActivation.png)



- 11 non-zero features using LeakyRelu with alpha as 0.01
- 32 using tanh
