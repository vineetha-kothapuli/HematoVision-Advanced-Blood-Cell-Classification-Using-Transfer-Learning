import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ✅ Path to filtered dataset
train_path = 'BCCD_Sorted'

# ✅ Check if dataset path is valid
if not os.path.exists(train_path):
    raise FileNotFoundError(f"❌ Dataset path not found: {train_path}")

# ✅ Create data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),  # MobileNetV2 prefers 224x224
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# ✅ Load MobileNetV2 without top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # 3 classes: WBC, RBC, Platelets

model = Model(inputs=base_model.input, outputs=predictions)

# ✅ Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# ✅ Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train the model
model.fit(train_data, validation_data=val_data, epochs=5)

# ✅ Save trained model
model.save('BloodCellClassifier.h5')
print("✅ Model saved as 'BloodCellClassifier.h5'")
