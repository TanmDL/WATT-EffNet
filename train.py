# initiate RMSprop optimizer

from keras import optimizers
opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)  # 0.0001, 1e-6

# Cosine Similarity
import tensorflow as tf

cosine_sim = tf.keras.losses.CosineSimilarity(axis=1)

########### Use code from MBonvblocks.py #####################################
efficient_net_b0.compile(loss= [cosine_sim, 'categorical_crossentropy'], loss_weights=[1.,1.] , optimizer=opt, metrics=[tf.keras.metrics.CosineSimilarity()])

history = efficient_net_b0.fit(train, validation_data= valid, epochs=300, batch_size = 10, verbose=1)
