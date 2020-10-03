"""
First try: Train a network to turn noise into (X,y) minimising loss function
"""
import tensorflow_probability as tfp
import tensorflow_transform as tft
#tfp.stats.covariance(xy_t)!=np.cov()
​
import sys
import numpy as np
import itertools
from itertools import combinations
​
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
import warnings
warnings.filterwarnings("ignore")
​
def make_random_data(n):
    """
    Make random Z=(X,y)=(x1,x2,x3,y) to feed the neural network
    """
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
​
    return x1, x2, x3, y
​
def t(x):
    return tf.transpose(x)
​
def calc_r2(x, y):
    if len(x.shape)==1:
        return 0
    det_C_xy = np.linalg.det(np.corrcoef(x.T, y))
    if x.shape[1]==0:
        det_C_x = 1
    else:
        det_C_x = np.linalg.det(np.corrcoef(x.T))
    return (1.0 - det_C_xy/det_C_x)
​
def tf_calc_r2(x, y):
    tensorshape = tf.shape(x)
    if len(tensorshape)==1: # Why 0?
        return 0.0
    yy = tf.expand_dims(y, axis=1)
    xy = tf.concat([x, yy], axis=1)
    det_C_xy = tf.linalg.det(tfp.stats.correlation(xy))     #np.linalg.det(np.corrcoef(x.T, y))
    if tensorshape[1]==0:
        det_C_x = 1.0
    else:
        det_C_x = tf.linalg.det(tfp.stats.correlation(x)) #np.linalg.det(np.corrcoef(x.T))
    return (1.0 - det_C_xy/det_C_x)
​
def tf_cov(x):
    x = x - tf.expand_dims(tf.reduce_mean(x, axis=1), 1)
    fact = tf.cast(tf.shape(x)[1] - 1, tf.float32)
    return tf.matmul(x, tf.math.conj(tf.transpose(x))) / fact
​
def r2_vs_cf(X, y):
    cf_dict = {():0, (0,):3, (1,):7, (2,):10, (0,1):7, (0,2):10, (1,2):10, (0,1,2):10}
    players = [0,1,2]
    #coalitions = list(itertools.chain(combinations(players,_i+1) for _i, _ in enumerate(players)))
    coalitions = list(itertools.chain(combinations(players,1),combinations(players,2),combinations(players,3)))
​
    sum_loss = 0
    for _coalition in coalitions:
        _x = tf.gather(X, _coalition, axis=1)
        sum_loss += tf_calc_r2(_x, y)-cf_dict[_coalition]
​
    return sum_loss/len(coalitions)
​
def test_loss():
    output = tf.constant([[0.093302466, 0, 0, 0.0539175197],
            [0, 0, 0.0205823872, 0.0604132339],
            [0, 0, 0.014283713, 0.0616389737],
            [0, 0.0603796616, 0.133852571, 0.353532702],
            [0, 0.123386495, 0.106639855, 0.571168661],
            [0.491044343, 0.0125491843, 0, 0.345430344]])
​
​
    target = tf.constant([[0.0390668213, 0.0617848709, 0.225083053, 0.190982759],
            [0, 0, 0.0564388037, 0.076279521],
            [0, 0, 0, 0],
            [0, 0.0207701698, 0.0687658116, 0.186252296],
            [0.042760849, 0.101211086, 0.25936386, 0.176173285],
            [0, 0, 0.0691789091, 0.0813835561]])
​
    a = tf.constant([[3.0,8.0],[4.0,6.0]])
​
    #x = target[:,:-1]
    #y = target[:,-1]
    #print(x.numpy())
    #print(y.numpy())
​
    x = output[:,:-1]
    y = output[:,-1]
    #print(tf_calc_r2(x, y))
    #print(r2_vs_cf(x, y))
​
    yy = tf.expand_dims(y, axis=1)
    xy = tf.concat([x, yy], axis=1)
​
    batch_x = tf.constant([1.0,3.0,3.0,4.0])
    batch_y = tf.constant(2.0)
    print(tf_calc_r2(batch_x, batch_y))
​
    # === OK:
    #output_np = output.numpy()
    #rtol = 1e-06
    #np.testing.assert_allclose(tf.linalg.det(a).numpy(),np.linalg.det(a.numpy()),rtol=rtol)
    #np.testing.assert_allclose(tf.linalg.det(a).numpy(),np.linalg.det(a.numpy()),rtol=rtol)
    #np.testing.assert_allclose(tf_cov(output).numpy(),np.cov(output_np),rtol=rtol)
    #np.testing.assert_allclose(tfp.stats.correlation(xy),np.corrcoef(x.numpy().T, y.numpy()),rtol=rtol)
    #np.testing.assert_allclose(tf_cov(output).numpy(),np.cov(output_np),rtol=rtol)
    #np.testing.assert_allclose(tf_calc_r2(x, y),calc_r2(x.numpy(),y.numpy()),rtol=rtol)
​
def make_network(input_dim):#, n_nodes):
    """
    Input:
        input_dim: Dimension of the input layer, corresponding to the number of
        features to autoencode
        n_nodes: List of node numbers for the layers
    Returns:
        model
    """
    assert isinstance(input_dim, int)
    #assert isinstance(n_nodes, list)
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(input_dim, activation="relu")) #usually softmax
​
    #input_layer = Input(shape=(input_dim, ), name="input")
    ## --- Put the first encoding layer on top of the input layer
    #encoding = Dense(n_nodes[0], activation=ACTIVATION_AE,
    #                 name="enc_0")(input_layer)
    ## --- Create all the encoding layers until the latent layer
    #for _n, _n_nodes in enumerate(n_nodes[1:n_middle], start=1):
    #    encoding = Dense(_n_nodes, activation=ACTIVATION_AE,
    #                     name="enc_{0}".format(_n))(encoding)
    ## --- Single output layer
    #decoded_output = Dense(input_dim, activation="linear",
    #                       name="out")(decoding)
    #return Model(input_layer, decoded_output)
​
    return model
​
​
def r2_loss(target, output):#Z_true, Z_pred):
    """
    Iputs are list of #batches lists containing network input/output resp
    """
​
​
    #for _batch in output:
    #    _x = output[:-1]
    #    _y = output[-1]
​
    #tf.print(tf.shape(output))
    #tf.print(output)
    #tf.print(output[0])
    #tf.print(output[0][:-1])
    #tf.print(output[0][-1])
    X = output[:,:-1]
    y = output[:,-1]
    #tf.print(X)
    #tf.print(y)
    tf.print(r2_vs_cf(X, y))
    return r2_vs_cf(X, y)
​
    output /=  tf.math.reduce_sum(output, axis=-1, keepdims=True)
    output = tf.clip_by_value(output, 1e-7, 1 - 1e-7)
    sample_loss = tf.math.reduce_sum(target * -tf.math.log(output), axis=-1, keepdims=False)
    #tf.print(sample_loss)
    #tf.print(tf.shape(sample_loss))
    #tf.print(tf.math.reduce_sum(sample_loss))
    #return tf.math.reduce_sum(sample_loss)
​
    #return sample_loss
​
    #x = output[:,:-1]
    #y = output[:,-1]
    #tf.print(tf_calc_r2(x, y))
    #tf.print(tf.shape(tf_calc_r2(x, y)))
​
​
if __name__ == "__main__":
​
    #test_loss()
    #sys.exit()
​
    # --- Data
    x0, x1, x2, y = make_random_data(100)
    X_train = np.vstack((x0, x1, x2, y)).T
​
    #r2_vs_cf(X_train, y)
    #sys.exit()
​
    # --- Data
    model = make_network(4)
    print(model.layers)
    print(model.summary())
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
​
    model.compile(loss=r2_loss, optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, X_train, epochs=2, batch_size=8)
​
    #test_loss()
    # --- shoot. what to put instead of the second X_train?
​
​
​
​
​
​
​
​
​
​
    # --- NOTES (probably doesnt work)
    #model.predict(random stuff) -> put that into r2 and compare to cf_dict
    #def tf_corrcoef(xy_t):
    #    cov_t = tf_cov(xy_t)
    #    cov2_t = tf.linalg.diag(1/tf.sqrt(tf.linalg.diag_part(cov_t)))
    #    return cov2_t @ cov_t @ cov2_t
    #
    #def test_tf():
    #    a = tf.constant([[1,2,3],[4,5,6]])
    #    proto_tensor = tf.make_tensor_proto(a)
    #    print(tf.make_ndarray(proto_tensor))
    #