############ Data Augmentation ##############

train_datagen =tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,vertical_flip=False)
valid_datagen =tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=False, vertical_flip=False)

train = train_datagen.flow(new_X,onehot_encoded,batch_size=16)
valid = valid_datagen.flow(new_X_valid,onehot_encodedvalid,batch_size=16)

######### initiate RMSprop optimizer #############

from keras import optimizers
opt = optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)  

############# Combined Loss ####################

class Total_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, y_true, y_pred):
      # Custom CE Loss
      log_y_pred = tf.keras.ops.log(y_pred)
      elements = -tf.math.multiply_no_nan(x=log_y_pred, y=y_true)
      CE_loss = tf.reduce_mean(tf.reduce_sum(elements,axis=1))
      # Cosine sim Loss
      cos_sim = -tf.reduce_sum(tf.nn.l2_normalize(y_true) * tf.nn.l2_normalize(y_pred), axis = -1)
      cossim = tf.reduce_mean(cos_sim)
      return 0.1*cossim + 0.9*CE_loss

################# Compile ########################

Watteffnet36.compile(loss = [Total_Loss], optimizer = opt, metrics = ['accuracy'])

################# Run Training ###########################
history = Watteffnet36.fit(train, validation_data= valid, epochs= 600, batch_size = 64, verbose=1)
