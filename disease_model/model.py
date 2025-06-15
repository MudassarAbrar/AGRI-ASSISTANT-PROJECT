# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from load_data import train_data, val_data


# # Number of output classes (change this according to your crop folders)
# NUM_CLASSES = 15 # For example: Tomato, Potato, Corn

# model = Sequential([
#     # First Convolutional Layer
#     Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     MaxPooling2D(pool_size=(2, 2)),

#     # Second Convolutional Layer
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     # Flatten to go from 2D to 1D
#     Flatten(),

#     # Fully Connected Layers
#     Dense(128, activation='relu'),
#     Dropout(0.5),  # Prevent overfitting
#     Dense(NUM_CLASSES, activation='softmax')  # Final output
# ])

# # Compile the model
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Print model summary
# model.summary()



# EPOCHS = 10  # You can increase this later for better results

# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=EPOCHS
# )




# model.save("plant_disease_model.h5")