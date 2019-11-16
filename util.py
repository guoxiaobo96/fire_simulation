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

