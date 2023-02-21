## Cosy ##

Cosy is a keras wrapper for soft-parameter sharing multitask learning.
The goal of this repository is to provide a simple high level wrapper for fast
prototyping of simple multitask learning problems. The repository ships with two 
wrappers, CosyNet and CosyNetMultiInput; both of these take an arbitrary list of
keras model configurations and implement a soft parameter sharing loss between the 
weights of the specified network layers. 

# Warning: #
This repository is still under development so use with caution


# --------------  Usage -------------- #

# Soft parameter sharing loss #
In soft parameter sharing one constrains the
weights of each network to one another whilst allowing for task
specific deviations. This is often done by appending the supervised loss
function of our network with a regularization loss that penalises our model parameters based on a distance
metric between model parameters. An equation demonstrating this process using the 
squared frobenius norm is shown below.

$L = l + \sum_{i,j \in N, i < j} \lambda_{ij} \lVert W_i - W_j \rVert^2$

In this equation, $N$ is the set of all neural networks, $i$ and $j$ are indices representing two different neural 
networks in $N$, and $\lambda_{ij}$ is a regularization parameter for the weight difference between the $i$th and $j$th 
neural networks. The sum is taken over all pairs of distinct neural networks in $N$.

For three neural networks connected (A, B and C) in a soft parameter sharing process this
equation can be defined as;

$L = l + \lambda_{AB} \lVert W_A - W_B \rVert^2 + \lambda_{AC} \lVert W_A - W_C \rVert^2 + \lambda_{BC} \lVert W_B - W_C \rVert^2$

This package currently implements six distance metrics for this process which can be imported 
from the loss module and are shown below

```python 
from cosy.losses import (
    squared_frobenius_norm,
    l1_norm,
    trace_norm,
    kl_divergence,
    mutual_information,
    wasserstein_distance,
)
```

# Wrappers #

The two wrapper can be imported from the models' module of cosy.
```python 
from cosy.models import CosyNet, CosyNetMultiInput
```
The CosyNet wrapper works with homogenous features sets. It takes 
one array for the training data and a nested array for the target data. An example of the
input data format for wrapping three networks together would be;

```python 
x = np.array(x)
y = np.array([y1, y2, y3])
```

The CosyNetMultiInput wrapper works with heterogeneous feature sets 
and has an input format of;

```python 
x = np.array(x1, x2, x3)
y = np.array([y1, y2, y3])
```

In order to train cosy net a set of model parameter must be specified,
these include a model configuration, which is inputted as a list of keras model
configuration, a min_layer_cutoff and a max_layer_cutoff which define the region
in which to implament soft parameter sharing, the soft parameter sharing loss_fn, and the 
scalar value, which can be a single value in the case of two network or a list of values 
for a greater number of networks. It is important that in the region  you are applying soft parameter sharing the layers have the same
shape and if you are using the mutli input model ensure the min_layer_cutoff is at least 1,
as the shapes of first layers will be different due to the shape of your feature set. 


```python 
import tensorflow as tf

input_ = tf.keras.Input(shape=(x_train.shape[1]))
x = tf.keras.layers.Dense(1000, activation='relu')(input_)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(500, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(1, activation='relu')(x)

model = tf.keras.Model(inputs=input_, outputs=x)

cosy_model = CosyNet(
    model_config=[model.get_config()] * 3,
    scalar=[0.3, 1.0, 0.2],
    max_layer_cutoff=-1,
    min_layer_cutoff=1,
    loss_fn=squared_frobenius_norm,
)
```

Once the cosy_model has been initialized it can be compiled and train in the same fashion
as a standard keras multitask network.

```python
import tensorflow_addons as tfa

opt = tf.keras.optimizers.Adam(learning_rate=0.00025)

losses = {
    'output_1': 'mean_squared_error',
    'output_2': 'mean_squared_error',
    'output_3': 'mean_squared_error',
    }

R2 = tfa.metrics.RSquare()

cosy_model.compile(
            optimizer=opt,
            loss=losses,
            metrics=[R2],
            )
```

# Custom callbacks # 
The reposity currently contain one custom callback, EarlyStoppingMultiLoss which monitors
a list of losses or metrics with stopping criteria and restores the best weights of each of the internal
models from the point at which the stopping criteria were met.

```python
from cosy.callbacks import EarlyStoppingMultiLoss

monitor_list_r2 = [
                   'val_output_1_r_square',
                   'val_output_2_r_square',
                   'val_output_3_r_square',
                ]

monitor_list_loss = [
                   'val_output_1_loss',
                   'val_output_2_loss',
                   'val_output_3_loss',
]

earlystopping = EarlyStoppingAtMinLoss(
                                       mode='max',
                                       monitor = monitor_list_r2,
                                       patience = 5
                                       )

earlystopping = EarlyStoppingAtMinLoss(
                                       mode='min',
                                       monitor = monitor_list_loss,
                                       patience = 5
                                       )


```

From here the model can be fit using the standard keras fit function. Here are examples
using CosyNet and CosyNetMultiInput.

```python
cosy_model.fit(
    x=x, y=[y1, y2, y3],
    validation_data=(x_val, [y1_val, y2_val, y3_val]),
    batch_size=32, epochs = 4, verbose=1
)

cosy_model_multi_input.fit( 
    x=[x1, x2, x3], y=[y1, y2, y3],
    validation_data=([x1_val, x2_val, x3_val], [y1_val, y2_val, y3_val]),
    batch_size=32, epochs = 4, verbose=1
)
```

## Masking missing data ##

An important aspect of soft parameter sharing multitask learning is the ability to co-train models with a different number of 
labelled data-point. In order to implement this workflow this repository includes masked supervised loss function (currently
only one but more coming soon). In order to train in this paradigm, specify the labels of the missing data points as -1 and use 
a masked loss function, this will remove these datapoints from the supervised loss whilst still implementing soft parameter loss
from the forward pass of the networks for these datapoints.

```python
from cosy.losses import MaskedMeanSquaredError

masked_loss = MaskedMeanSquaredError()
```











