import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def jacobian(x):
    dudx = x[:,:,1:,0] - x[:,:,:-1,0]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dvdy = x[:,1:,:,1] - x[:,:-1,:,1]
    
    dudx = tf.concat([dudx,tf.expand_dims(dudx[:,:,-1], axis=2)], axis=2)
    dvdx = tf.concat([dvdx,tf.expand_dims(dvdx[:,:,-1], axis=2)], axis=2)
    dudy = tf.concat([dudy,tf.expand_dims(dudy[:,-1,:], axis=1)], axis=1)
    dvdy = tf.concat([dvdy,tf.expand_dims(dvdy[:,-1,:], axis=1)], axis=1)

    j = tf.stack([dudx,dudy,dvdx,dvdy], axis=-1)
    w = tf.expand_dims(dvdx - dudy, axis=-1) # vorticity (for visualization)
    return j, w

def curl(x):
    u = x[:,1:,:,0] - x[:,:-1,:,0] # ds/dy
    v = x[:,:,:-1,0] - x[:,:,1:,0] # -ds/dx,
    u = tf.concat([u, tf.expand_dims(u[:,-1,:], axis=1)], axis=1)
    v = tf.concat([v, tf.expand_dims(v[:,:,-1], axis=2)], axis=2)
    c = tf.stack([u,v], axis=-1)
    return c


def get_tensor_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else - 1 for num in shape]

def add_channels(x, num_ch=1):
    b, h, w, c = get_tensor_shape(x)
    x = tf.concat([x, tf.zeros([b, h, w, num_ch])], axis=-1)
    return x

def remove_channels(x):
    b, h, w, c = get_tensor_shape(x)
    x, _ = tf.split(x, [3, -1], axis=3)
    return x

def denorm_img(norm):
    _, _, _, c = get_tensor_shape(norm)
    if c == 2:
        norm = add_channels(norm, num_ch=1)
    elif c > 3:
        norm = remove_channels(norm)
    img = tf.cast(tf.clip_by_value((norm + 1)*127.5, 0, 255), tf.uint8)
    return img