import numpy as np
import tensorflow as tf

def spectral_normed_weight(w, 
    u=None, 
    num_iters=1, # For Power iteration method, usually num_iters = 1 will be enough
    update_collection=None, 
    with_sigma=False # Estimated Spectral Norm
    ):

    w_shape = w.shape.as_list()
    w_new_shape = [ np.prod(w_shape[:-1]), w_shape[-1] ]
    w_reshaped = tf.reshape(w, w_new_shape, name='w_reshaped')
    
    if u is None:
        u = tf.get_variable("u_vec", [w_new_shape[0], 1], initializer=tf.random_normal_initializer(), trainable=False)
    
    # power iteration
    u_ = u
    for _ in range(num_iters):
        # ( w_new_shape[1], w_new_shape[0] ) * ( w_new_shape[0], 1 ) -> ( w_new_shape[1], 1 )
        v_ = _l2normalize(tf.matmul(tf.transpose(w_reshaped), u_)) 
        # ( w_new_shape[0], w_new_shape[1] ) * ( w_new_shape[1], 1 ) -> ( w_new_shape[0], 1 )
        u_ = _l2normalize(tf.matmul(w_reshaped, v_))

    u_final = tf.identity(u_, name='u_final') # ( w_new_shape[0], 1 )
    v_final = tf.identity(v_, name='v_final') # ( w_new_shape[1], 1 )

    u_final = tf.stop_gradient(u_final)
    v_final = tf.stop_gradient(v_final)

    sigma = tf.matmul(tf.matmul(tf.transpose(u_final), w_reshaped), v_final, name="est_sigma")

    update_u_op = tf.assign(u, u_final)

    with tf.control_dependencies([update_u_op]):
        sigma = tf.identity(sigma)
        w_bar = tf.identity(w / sigma, 'w_bar')

    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def _l2normalize(v, eps=1e-12):
    with tf.name_scope('l2normalize'):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def conv2d(inputs, 
    filters, k_size, strides=1,
    padding="SAME",
    use_bias=True, 
    spectral_normed=True, 
    name="conv2d"
    ):

    with tf.variable_scope(name):

        w = tf.get_variable("w_{}".format(name), 
            shape=[k_size, k_size, inputs.get_shape()[-1], filters], 
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k_size * k_size * filters))
            )
        
        if spectral_normed:
            w = spectral_normed_weight(w)
        
        conv = tf.nn.conv2d(inputs, w, strides=strides, padding=padding.upper(), name="conv_w_{}".format(name))
        
        if use_bias:
            biases = tf.get_variable("b_{}".format(name), [filters], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases, name="conv_add_b_{}".format(name))
        
        return conv

'''
def linear(inputs, 
    out_dim, 
    w_init=None,
    activation=None,
    use_bias=True, bias_start=0.0,
    spectral_normed=False, 
    name="linear", 
    ):

    with tf.variable_scope(name):

        w = tf.get_variable("w", 
            shape=[ inputs.get_shape()[-1], out_dim ], 
            dtype=tf.float32,
            initializer=w_init
            )
        
        if spectral_normed:
            w = spectral_normed_weight(w)

        mul = tf.matmul(inputs, w, name='linear_mul')

        if use_bias:
            bias = tf.get_variable("b", [out_dim], initializer=tf.constant_initializer(bias_start))
            mul = tf.nn.bias_add(mul, bias, name="mul_add_b")

        if not (activation is None):
            mul = activation(mul)

        return mul
'''