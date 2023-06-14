#As discussed, Please find the below usecase

#Use case : Describe an Image

 

#Requirement: Build a model to describe the content of the given image using properly formed English sentences. A minimum of 2 descriptions per image is to be generated.

#Data Set: Dataset & its properties are listed in the link below.

                
#https://github.com/BryanPlummer/flickr30k_entities
# Buiding a model

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

# Set the paths to the dataset and annotations
dataset_dir = "path/to/your/flickr30k_dataset"
annotation_file = "path/to/your/annotations_file.csv"

# Set the random seed for reproducibility
random.seed(42)

# Load the annotations
annotations = pd.read_csv(annotation_file, delimiter="|")
image_ids = annotations["image_id"]
image_paths = annotations["image_path"]
captions = annotations["caption"]

# Preprocess the captions
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_caption(caption):
    caption = caption.lower()
    tokens = word_tokenize(caption)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_caption = " ".join(tokens)
    return preprocessed_caption

preprocessed_captions = [preprocess_caption(caption) for caption in captions]

# Create a vocabulary
vocab = set()
for caption in preprocessed_captions:
    vocab.update(caption.split())

vocab_size = len(vocab)
word_to_idx = {word: idx+1 for idx, word in enumerate(vocab)}  # Assign index 0 for padding
idx_to_word = {idx+1: word for idx, word in enumerate(vocab)}  # Assign index 0 for padding

# Load the VGG16 model without the top layer
base_model = VGG16(weights="imagenet", include_top=False)

# Define the image captioning model
def build_model(embed_size, hidden_size, vocab_size):
    # Define the image model
    image_input = Input(shape=(224, 224, 3))
    image_features = base_model(image_input)
    image_features = tf.keras.layers.Flatten()(image_features)
    image_model = Model(inputs=image_input, outputs=image_features)
    
    # Define the caption model
    caption_input = Input(shape=(None,))
    caption_embedding = Embedding(vocab_size+1, embed_size)(caption_input)
    caption_lstm = LSTM(hidden_size)(caption_embedding)
    caption_model = Model(inputs=caption_input, outputs=caption_lstm)
    
    # Concatenate the image and caption models
    combined = tf.keras.layers.concatenate([image_model.output, caption_model.output])
    dense = Dense(256, activation="relu")(combined)
    output = Dense(vocab_size+1, activation="softmax")(dense)
    
    # Create the model
    model = Model(inputs=[image_model.input, caption_model.input], outputs=output)
    return model

# Set the hyperparameters
embed_size = 256
hidden_size = 256
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Build the model
model = build_model(embed_size, hidden_size, vocab_size)
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate))

# Define the data generator
def data_generator(image_paths, captions, word_to_idx, batch_size):
    num_samples = len(image_paths)
    while True:
        indices = random.sample(range(num_samples), batch_size)
        batch_image_paths = [image_paths[idx] for idx in indices]
        batch_captions = [captions[idx] for idx in indices]
        batch_images = []
        batch_sequences = []
        batch_targets = []
        for image_path, caption in zip(batch_image_paths, batch_captions):
            # Load and preprocess the image
            img = image.load_img(os.path.join(dataset_dir, image_path), target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.vgg16.preprocess_input(img)
            batch_images.append(img)
            
            # Preprocess the caption
            preprocessed_caption = preprocess_caption(caption)
            sequence = [word_to_idx[word] for word in preprocessed_caption.split()]
            sequence = pad_sequences([sequence], padding="post")[0]
            target = tf.keras.utils.to_categorical(sequence[1:], num_classes=vocab_size+1)
            batch_sequences.append(sequence)
            batch_targets.append(target)
        
        batch_images = np.concatenate(batch_images, axis=0)
        batch_sequences = np.array(batch_sequences)
        batch_targets = np.array(batch_targets)
        yield ([batch_images, batch_sequences], batch_targets)

# Train the model
num_steps = len(image_paths) // batch_size
train_generator = data_generator(image_paths, preprocessed_captions, word_to_idx, batch_size)
model.fit(train_generator, steps_per_epoch=num_steps, epochs=num_epochs)

# Generate captions for test images
test_image_paths = image_paths[-10:]  # Get the last 10 images as test images
test_images = []
for image_path in test_image_paths:
    img = image.load_img(os.path.join(dataset_dir, image_path), target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    test_images.append(img)

test_sequences = [["<start>"]] * len(test_image_paths)
max_caption_length = 50

for _ in range(max_caption_length):
    next_words = []
    for sequence in test_sequences:
        sequence_idxs = [word_to_idx[word] for word in sequence]
        sequence_idxs = pad_sequences([sequence_idxs], padding="post")
        predicted_idxs = np.argmax(model.predict([np.array(test_images), sequence_idxs]), axis=2)
        predicted_words = [idx_to_word[idx] for idx in predicted_idxs[0]]
        next_words.append(predicted_words[-1])
    test_sequences = [sequence + [word] for sequence, word in zip(test_sequences, next_words)]

# Print the generated captions for test images
for i, sequence in enumerate(test_sequences):
    caption = " ".join(sequence[1:-1])  # Remove <start> and <end> tokens
    print("Image:", test_image_paths[i])
    print("Generated Caption:", caption)
    print()
    
    
#Make sure to replace "path/to/your/flickr30k_dataset" with the actual path to the Flickr30k dataset directory and "path/to/your/annotations_file.csv" with the path to the annotations file in CSV format. Also, ensure that you have the necessary dependencies installed (e.g., TensorFlow, NLTK) and the required NLTK resources downloaded.

#This code uses the VGG16 model as a feature extractor for images and combines it with an LSTM-based caption model. It preprocesses the captions by tokenizing, lemmatizing, and removing stopwords. The model is trained using a data generator that loads images and captions in batches. Finally, it generates captions for a set of test images using the trained model.

