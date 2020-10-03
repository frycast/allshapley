"""
This is just me working my way through the keras tutorial
for researchers: 
https://keras.io/getting_started/intro_to_keras_for_researchers/

I've read up to the heading 'Layer gradients'
"""

# %%
import tensorflow as tf
from tensorflow import keras

# %%
x = tf.constant([[5, 2], [1, 3]])
print(x)

# %%
x.numpy()

# %%
print("dtype:", x.dtype)
print("shape:", x.shape)

# %%
print(tf.ones(shape=(2, 1)))
print(tf.zeros(shape=(2, 1)))

# %%
x = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)
x

# %%
x = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype="int32")
x

"""
Variables are special tensors used to store 
mutable state (like the weights of a neural network).
"""
# %%
initial_value = tf.random.normal(shape=(2, 2))
a = tf.Variable(initial_value)
print(a)

"""
You update the value of a Variable by using the 
methods .assign(value), .assign_add(increment), 
or .assign_sub(decrement)
"""
# %%
new_value = tf.random.normal(shape=(2, 2))
a.assign(new_value)
for i in range(2):
    for j in range(2):
        assert a[i, j] == new_value[i, j]
print(a)

# %%
added_value = tf.random.normal(shape=(2, 2))
print(added_value)
a.assign_add(added_value)
for i in range(2):
    for j in range(2):
        assert a[i, j] == new_value[i, j] + added_value[i, j]
print(a)

"""
If you've used NumPy, doing math in TensorFlow will look very familiar. 
The main difference is that your TensorFlow code can run on GPU and TPU.
"""
# %%
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

c = a + b
d = tf.square(c)
e = tf.exp(d)

"""
Gradients
---------
You can automatically retrieve the gradient 
of any differentiable expression.

Just open a GradientTape, start "watching" 
a tensor via tape.watch(), and compose a 
differentiable expression using this 
tensor as input:
"""
# %%
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

with tf.GradientTape() as tape:
    tape.watch(a)  # Start recording the history of operations applied to `a`
    c = tf.sqrt(tf.square(a) + tf.square(b))  # Do some math using `a`
    # What's the gradient of `c` with respect to `a`?
    dc_da = tape.gradient(c, a)
    print(dc_da)

"""
By default, variables are watched automatically, 
so you don't need to manually watch them:
"""
# %%
a = tf.Variable(a)

with tf.GradientTape() as tape:
    c = tf.sqrt(tf.square(a) + tf.square(b))
    dc_da = tape.gradient(c, a)
    print(dc_da)

"""
Note that you can compute higher-order 
derivatives by nesting tapes:
"""
# %%
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as tape:
        c = tf.sqrt(tf.square(a) + tf.square(b))
        dc_da = tape.gradient(c, a)
    d2c_da2 = outer_tape.gradient(dc_da, a)
    print(d2c_da2)