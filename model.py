import tensorflow as tf
import numpy as np
import pandas as pd

df_strains = pd.read_csv('training_data.csv')
df_labels = pd.read_csv('training_labels.csv')

td_strains = pd.read_csv('testing_data.csv')
td_effects = pd.read_csv('testing_labels.csv')

terpene_train_data = np.array(df_strains)
effect_training_labels = np.array(df_labels)

terpene_test_data = np.array(td_strains)
effect_testing_labels = np.array(td_effects)

print(terpene_train_data.shape)
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(13, activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to the training data
model.fit(terpene_train_data, effect_training_labels, epochs=5)

# Evaluate the model on the test data
#test_loss, test_acc = model.evaluate(terpene_test_data, effect_testing_labels)
#print('Test accuracy:', test_acc)

# Make predictions on the test set
predictions = model.predict(terpene_test_data)

# Convert the predictions to class labels
class_labels = np.argmax(predictions, axis=1) #THIS IS INCORRECT

# Iterate through the test set and print the predictions and true labels
for i, (class_labels, true_label) in enumerate(zip(predictions, effect_testing_labels)):
    print(f"Sample {i}: Prediction: {class_labels}, True label: {true_label}")
