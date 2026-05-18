#I built two models for the project: a custom CNN model and a MobileNetV2 transfer learning model.
#The custom CNN uses convolution, pooling, dense, and dropout layers.
#The MobileNetV2 model uses a pre-trained backbone with some frozen layers and a custom classification head.
#I also created a binary version of MobileNetV2 for two-class classification.

#I imported MobileNetV2, which is a pre-trained deep learning model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import ( #I imported the layers needed to build CNN and MobileNetV2-based models
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential, Model #I imported Sequential for simple CNN and Model for custom MobileNetV2 architecture



# CUSTOM CNN MODEL
# I created a function to build a custom CNN model with 4 output classes
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4):

    model = Sequential() #I started a Sequential model

    # 1st Convolution Block
    model.add( #I added the first convolution layer with 32 filters to detect basic image features
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=input_shape
        )
    )

    model.add(MaxPooling2D((2, 2))) #I added max pooling to reduce image size and keep important features

    # 2nd Convolution Block
    model.add( #I added a second convolution layer with 64 filters for deeper feature extraction
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPooling2D((2, 2))) #I reduced the size again using max pooling

    # 3rd Convolution Block
    model.add( #I added a third convolution layer with 128 filters for more advanced feature learning
        Conv2D(
            128,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPooling2D((2, 2))) #I reduced the feature map size again

    # Flatten(I converted the 2D feature maps into a 1D vector)
    model.add(Flatten())

    # Dense Layer(I added a dense layer to learn high-level patterns)
    model.add(Dense(128, activation="relu"))

    # Dropout(I added dropout to reduce overfitting)
    model.add(Dropout(0.5))

    # Output Layer(I added the final output layer with softmax to predict 4 classes)
    model.add(Dense(num_classes, activation="softmax"))

    return model #I returned the custom CNN model



# MOBILENETV2 MODEL
# I created a function to build a MobileNetV2-based model
def build_mobilenetv2(
    input_shape=(224, 224, 3),
    num_classes=4
):

    # Pretrained Base Model(I loaded the pre-trained MobileNetV2 model without its final classification layer)
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze pretrained layers(I allowed the base model to be trainable)
    base_model.trainable = True

    for layer in base_model.layers[:-30]: #I froze most of the early layers and kept only the last 30 layers trainable
        layer.trainable = False

    # Custom Head
    x = base_model.output #I took the output from the base model

    x = GlobalAveragePooling2D()(x) #I converted feature maps into a compact vector

    x = Dense(128, activation="relu")(x)  #I added a dense layer for classification learning

    x = Dropout(0.5)(x) #I added dropout to reduce overfitting

    output = Dense(
        num_classes,
        activation="softmax"
    )(x) #I created the final output layer for 4-class prediction

    model = Model(
        inputs=base_model.input,
        outputs=output
    )  #I connected the base model and custom head to create the final model

    return model



# BINARY MODEL
# I created a function for binary classification
def build_binary_mobilenetv2():

    return build_mobilenetv2(num_classes=2) #I reused the MobileNetV2 model but changed the output to 2 classes