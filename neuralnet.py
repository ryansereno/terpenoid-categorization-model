import tensorflow as tf
import numpy as np
import pandas as pd

df_strains = pd.read_csv('training_data.csv')
df_labels = pd.read_csv('training_labels.csv')

terpene_train_data = np.array(df_strains)
effect_training_labels = np.array(df_labels)

print(terpene_train_data.shape)
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(13, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to the training data
model.fit(terpene_train_data, effect_training_labels, epochs=10, verbose='2')

# Evaluate the model on the test data
#test_loss, test_acc = model.evaluate(X_test, y_test)
#print('Test accuracy:', test_acc)
