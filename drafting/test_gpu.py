import tensorflow as tf

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

tf.debugging.set_log_device_placement(True)

a = tf.constant([ [1.0,2.0,3.0],[4.0,5.0,6.0] ])
b = tf.constant([ [1.0,2.0],[3.0,4.0],[5.0,6.0]])
c = a @ b

print(c)