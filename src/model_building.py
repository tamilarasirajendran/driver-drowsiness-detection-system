
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential, Model


# =====================================================
# CUSTOM CNN MODEL
# =====================================================
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4):

    model = Sequential()

    # 1st Convolution Block
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=input_shape
        )
    )

    model.add(MaxPooling2D((2, 2)))

    # 2nd Convolution Block
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPooling2D((2, 2)))

    # 3rd Convolution Block
    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPooling2D((2, 2)))

    # Flatten
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(128, activation="relu"))

    # Dropout
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation="softmax"))

    return model


# =====================================================
# MOBILENETV2 MODEL
# =====================================================
def build_mobilenetv2(
    input_shape=(224, 224, 3),
    num_classes=4
):

    # Pretrained Base Model
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze pretrained layers
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Custom Head
    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation="relu")(x)

    x = Dropout(0.5)(x)

    output = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    return model


# =====================================================
# BINARY MODEL
# =====================================================
def build_binary_mobilenetv2():

    return build_mobilenetv2(num_classes=2)