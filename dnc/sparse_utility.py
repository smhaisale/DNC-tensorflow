import tensorflow as tf

def sparse_zeros_3D(x,y,z):
    return tf.SparseTensor(indices=[[0,0,0]],values=[0.0], dense_shape=[x, y,z])

def sparse_zeros_2D(x,y):
    return tf.SparseTensor(indices=[[0,0]], values=[0.0], dense_shape=[x,y,])

def dense_to_sparse(a):
    idx = tf.where(tf.not_equal(a, 0))
    sparse = tf.SparseTensor(idx, tf.gather_nd(a, idx), a.get_shape())
    return sparse

def force_sparse_top_k_3D(a, k=8):
    m = tf.squeeze(a,0)
    m = tf.transpose(m)
    values, indices = tf.nn.top_k(m, k=8)
    sparse_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
    sparse_range = tf.tile(sparse_range, [1, k])
    full_indices = tf.concat(axis=2, values=[tf.expand_dims(sparse_range, 2), tf.expand_dims(indices, 2)])
    full_indices = tf.reshape(full_indices, [-1, 2])
    s = tf.sparse_to_dense(full_indices, m.get_shape(), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
    s = tf.transpose(s)
    s = tf.expand_dims(s, 0)
    return s


    
