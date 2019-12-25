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
