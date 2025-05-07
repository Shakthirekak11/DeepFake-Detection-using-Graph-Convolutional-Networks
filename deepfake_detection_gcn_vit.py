#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 as cv
import glob
import tensorflow
from tensorflow import keras


# In[ ]:


import zipfile
import os

# Path to the zip file in Downloads folder
zip_path = 'C:/Users/Shakthireka/Downloads/deepfake-detection-challenge.zip'

# Destination folder for extraction
extract_dir = 'C:/Users/Shakthireka/Downloads/DL PROJECT FINAL/extracted'

# Check if the extracted folder already exists
if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
    # If folder doesn't exist or is empty, extract the files
    os.makedirs(extract_dir, exist_ok=True)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Unzipping complete! Files are saved in '{extract_dir}'")
else:
    print(f"The directory '{extract_dir}' already exists and is not empty. Skipping extraction.")


# In[ ]:


import os

# Check current working directory
print("Current working directory:", os.getcwd())

# List all files in the 'Downloads' directory
downloads_dir = 'Downloads'
if os.path.exists(downloads_dir):
    print("Files in Downloads folder:", os.listdir(downloads_dir))
else:
    print("Downloads directory does not exist.")


# In[ ]:


train_sample_metadata = pd.read_json('extracted/train_sample_videos/metadata.json').T
train_sample_metadata.head(20)



# In[ ]:


train_sample_metadata.info()


# In[ ]:


real_videos = train_sample_metadata[train_sample_metadata.label == 'REAL']
fake_videos = train_sample_metadata[train_sample_metadata.label == 'FAKE']


# In[ ]:


real_videos.head(20)


# In[ ]:


fake_videos.head(20)


# Index is the Name of the Video
# Label represents the type of video it is.
# if the label is Fake orginal column contains its orginal video name else it is None

# In[ ]:


train_sample_metadata.info()


# In[ ]:


train_sample_metadata['label'].value_counts()


# In[ ]:


train_sample_metadata.groupby('label')['label'].count().plot(figsize=(10, 5), kind='bar', title='Distribution of Labels in the Training Set')
plt.show()


# In[ ]:


from IPython.display import Video
print("REAL VIDEO SAMPLE")
# Assume the videos are stored in a directory (for example, /content/drive/MyDrive/videos/)
video_path = f'extracted/train_sample_videos/aelfnikyqj.mp4'  # path to the first video in the list

# Display the video in Colab
Video(video_path, embed=True, width=600, height=400)


# In[ ]:


from IPython.display import Video
print("FAKE VIDEO SAMPLE")
# Assume the videos are stored in a directory (for example, /content/drive/MyDrive/videos/)
video_path = f'extracted/train_sample_videos/aagfhgtpmv.mp4'  # path to the first video in the list

# Display the video in Colab
Video(video_path, embed=True, width=600, height=400)


# In[ ]:


import os
import shutil

# Path where the 'real' directory will be created
real_folder_path = 'C:/Users/Shakthireka/Downloads/DL PROJECT FINAL/real'

# Check if the 'real' folder already exists
if os.path.exists(real_folder_path) and os.listdir(real_folder_path):
    print("The 'real' folder already exists and contains videos. Real videos are saved.")
else:
    # Create the 'real' folder if it doesn't exist
    os.makedirs(real_folder_path, exist_ok=True)

    # Path to the folder where your videos are located
    video_folder = 'extracted/train_sample_videos'

    # Filter the REAL videos from the DataFrame (train_sample_metadata)
    real_videos = train_sample_metadata[train_sample_metadata['label'] == 'REAL'].index

    # Loop through the list of real videos and copy them to the 'real' folder
    for video in real_videos:
        video_path = os.path.join(video_folder, video)

        # Check if the video exists before copying
        if os.path.exists(video_path):
            shutil.copy(video_path, real_folder_path)
        else:
            print(f"Video {video} not found in {video_folder}")

    print("All REAL videos have been copied to the 'real' folder.")


# In[ ]:


import os
import shutil


fake_folder_path = 'C:/Users/Shakthireka/Downloads/DL PROJECT FINAL/fake'


if os.path.exists(fake_folder_path) and os.listdir(fake_folder_path):
    print("The 'fake' folder already exists and contains videos. Fake videos are saved.")
else:
    os.makedirs(fake_folder_path, exist_ok=True)

    # Path to the folder where your videos are located
    video_folder = 'extracted/train_sample_videos'


    fake_videos = train_sample_metadata[train_sample_metadata['label'] == 'FAKE'].index

    for video in fake_videos:
        video_path = os.path.join(video_folder, video)

        # Check if the video exists before copying
        if os.path.exists(video_path):
            shutil.copy(video_path, fake_folder_path)
        else:
            print(f"Video {video} not found in {video_folder}")

    print("All FAKE videos have been copied to the 'fake' folder.")


# In[ ]:


videos = glob.glob('real/*.mp4')
frame_cnt = []
for video in videos:
  capture = cv.VideoCapture(video)
  frame_cnt.append(int(capture.get(cv.CAP_PROP_FRAME_COUNT)))
print("Frames: ",frame_cnt)
print("Avg Frame per video: ",np.mean(frame_cnt))


# In[ ]:


# Function to display frames from a video
def display_video_frames(video_path, num_frames=5):
    capture = cv.VideoCapture(video_path)
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))

    for i, idx in enumerate(frame_indices):
        capture.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            axes[i].imshow(frame)
            axes[i].axis('off')

    capture.release()
    plt.show()

# Display frames from a sample real video
print("Sample frames from a REAL video:")
display_video_frames('C:/Users/Shakthireka/Downloads/DL PROJECT FINAL/real/abarnvbtwb.mp4')

# Display frames from a sample fake video
print("Sample frames from a FAKE video:")
display_video_frames('C:/Users/Shakthireka/Downloads/DL PROJECT FINAL/fake/aagfhgtpmv.mp4')


# In[ ]:


def image_from_video(video_path):
    '''
    input: video_path - path for video
    process:
    1. perform a video capture from the video
    2. read the image
    3. display the image
    '''
    capture_image = cv.VideoCapture(video_path)
    ret, frame = capture_image.read()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)   # converting the frame color to RGB
    ax.imshow(frame)

    return frame


# In[ ]:


image = image_from_video("real/abarnvbtwb.mp4")
image


# In[ ]:


get_ipython().system('pip install cMake')


# In[ ]:


# Install necessary libraries
get_ipython().system('pip install torch torchvision')


# In[ ]:


get_ipython().system('pip install facenet_pytorch')


# In[ ]:


get_ipython().system('pip install --upgrade ipywidgets')


# In[ ]:


import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# In[ ]:


import os
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Define paths
real_videos_path = 'real/*.mp4'
fake_videos_path = 'fake/*.mp4'
train_sample_metadata_path = 'extracted/train_sample_videos/metadata.json'

# Load metadata
train_sample_metadata = pd.read_json(train_sample_metadata_path)

# Load face detection model
face_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
face_detection_model = face_detection_model.eval().to(device)

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', num_classes=2, device=device).eval()


# In[ ]:


# Transform for the input images
transform = transforms.Compose([
    transforms.ToTensor(),
])

def detect_faces(image, model, device):
    image_tensor = transform(image).to(device)
    with torch.no_grad():
        detections = model([image_tensor])[0]
    return detections

def extract_faces(image, detections, threshold=0.5):
    faces = []
    for i in range(len(detections['boxes'])):
        if detections['scores'][i] >= threshold:
            box = detections['boxes'][i].cpu().numpy().astype(int)
            face = image.crop((box[0], box[1], box[2], box[3]))
            faces.append(face)
    return faces


# In[ ]:


# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def extract_frames(video_path, num_frames=20):
    """
    Extracts evenly spaced frames from a video.
    """
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for idx in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    capture.release()

    return frames

def extract_embeddings(frames):
    """
    Extract embeddings for each frame using InceptionResnetV1.
    """
    embeddings = []
    for frame in frames:
        input_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(input_tensor).squeeze().cpu().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)


# In[ ]:


import os
import pickle
from glob import glob

# Define paths for video embeddings and labels
embeddings_file_path = 'video_embeddings.pkl'
labels_file_path = 'video_labels.pkl'

# Check if embeddings and labels already exist
if os.path.exists(embeddings_file_path) and os.path.exists(labels_file_path):
    print("Embeddings and labels already exist. Skipping extraction.")
    with open(embeddings_file_path, 'rb') as f:
        video_embeddings = pickle.load(f)
    with open(labels_file_path, 'rb') as f:
        video_labels = pickle.load(f)
else:
    # Correct the path usage with glob
    real_videos = sorted(glob('real/*.mp4'))[:70]
    fake_videos = sorted(glob('fake/*.mp4'))[:70]

    video_files = real_videos + fake_videos
    labels = [0] * len(real_videos) + [1] * len(fake_videos)  # 0 for REAL, 1 for FAKE

    # Initialize lists to store results
    video_embeddings = []
    video_labels = []

    for i, video_file in enumerate(video_files):
        print(f'Processing video {i + 1}/{len(video_files)}: {video_file}')

        # Extract frames and embeddings
        frames = extract_frames(video_file, num_frames=20)
        embeddings = extract_embeddings(frames)

        video_embeddings.append(embeddings)
        video_labels.append(labels[i])

    # Save embeddings and labels as pickle files in the current directory
    with open(embeddings_file_path, 'wb') as f:
        pickle.dump(video_embeddings, f)
    with open(labels_file_path, 'wb') as f:
        pickle.dump(video_labels, f)

    print("Extraction complete and data saved.")


# In[ ]:


import os
import pickle
import matplotlib.pyplot as plt

# Path to the embeddings file
embeddings_file_path = 'video_embeddings.pkl'

def plot_embeddings(embeddings, video_idx=0):
    """
    Visualize embeddings for a given video index.
    """
    video_embed = embeddings[video_idx]
    plt.figure(figsize=(10, 6))
    plt.imshow(video_embed.T, aspect='auto')
    plt.colorbar()
    plt.title(f'Embeddings for video {video_idx}')
    plt.show()

# Load embeddings from the pickle file
if os.path.exists(embeddings_file_path):
    with open(embeddings_file_path, 'rb') as f:
        loaded_embeddings = pickle.load(f)

    # Plot embeddings for specified video indices
    plot_embeddings(loaded_embeddings, video_idx=0)
    plot_embeddings(loaded_embeddings, video_idx=130)
else:
    print(f"File '{embeddings_file_path}' not found. Please check the path.")


# In[ ]:


# Import necessary libraries for GCN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split


# In[ ]:


def create_graph(embeddings, threshold=0.9):
    """
    Create a graph for video embeddings where nodes are frames, and edges are based on cosine similarity.
    """
    num_nodes = embeddings.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Create an adjacency matrix based on cosine similarity
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            if similarity > threshold:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Undirected graph

    return adj_matrix

def graph_from_embeddings(embeddings, threshold=0.9):
    """
    Create a PyTorch geometric data object from video embeddings.
    """
    # Get number of nodes (frames)
    num_nodes = embeddings.shape[0]

    # Create adjacency matrix
    adj_matrix = create_graph(embeddings, threshold=threshold)

    # Create edge index (PyTorch geometric format)
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create node features
    x = torch.tensor(embeddings, dtype=torch.float)

    # Create graph data object
    data = Data(x=x, edge_index=edge_index)

    return data


# In[ ]:


# Function to display video frames
def display_video_frames(video_path, num_frames=5):
    capture = cv.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while capture.isOpened() and frame_count < num_frames:
        ret, frame = capture.read()
        if not ret:
            break
        frame_count += 1
        frames.append(frame)
    capture.release()

    # Display the frames using matplotlib
    fig, axs = plt.subplots(1, len(frames), figsize=(20, 20))
    for i, frame in enumerate(frames):
        axs[i].imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        axs[i].axis('off')
    plt.show()

# Function to plot the graph
def plot_graph(graph_data):
    edge_index = graph_data.edge_index.cpu().numpy()
    num_nodes = graph_data.num_nodes

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.T)

    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.show()

# Modified function to visualize steps and predict
def visualize(video_path, model, device, threshold=0.9):
    print(f"Processing video: {video_path}")

    # Display the video frames
    print("Displaying the video frames:")
    display_video_frames(video_path)

    # Extract embeddings and graph data
    frames = extract_frames(video_path, num_frames=20)
    embeddings = extract_embeddings(frames)
    graph_data = graph_from_embeddings(embeddings, threshold=threshold).to(device)

    # Plot the graph
    print("Plotting the graph:")
    plot_graph(graph_data)
# List of video paths to process
video_paths = [
    'C:/Users/Shakthireka/Downloads/DL PROJECT FINAL/extracted/train_sample_videos/axntxmycwd.mp4'
]

# Loop through the video paths and visualize & predict each
for video_path in video_paths:
    visualize(video_path, model, device)


# In[ ]:


graphs = []
for embeddings in video_embeddings:
    graph_data = graph_from_embeddings(embeddings)
    graphs.append(graph_data)

# Convert labels to tensor
labels = torch.tensor(video_labels, dtype=torch.long)

# Split into training and testing sets (you can adjust the split ratio as needed)
train_graphs, test_graphs, train_labels, test_labels = train_test_split(graphs, labels, test_size=0.2, random_state=42)


# In[ ]:


class GCNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply the first GCN layer and ReLU activation
        x = F.relu(self.conv1(x, edge_index))

        # Apply the second GCN layer
        x = self.conv2(x, edge_index)

        # Global pooling (mean) over all nodes to get graph-level output
        x = torch.mean(x, dim=0)

        return x


# In[ ]:


import matplotlib.pyplot as plt

# Define model parameters
input_dim = video_embeddings[0].shape[1]  # Number of features (embedding size)
hidden_dim = 64  # Adjust as necessary
output_dim = 2   # Since we have two classes: REAL and FAKE

# Initialize the model, optimizer, and loss function
model = GCNNet(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as necessary
criterion = nn.CrossEntropyLoss()

# Initialize lists to store loss and accuracy for each epoch
loss_values = []
accuracy_values = []

# Training loop
num_epochs = 300  # Adjust the number of epochs as needed
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for data, label in zip(train_graphs, train_labels):
        optimizer.zero_grad()

        # Ensure data is in the right format for the model
        data = data.to(device)
        label = label.to(device)

        # Forward pass
        output = model(data)

        # Check if output and label are of the correct shape
        if output.dim() > 1:
            output = output.view(-1)  # Flatten if necessary
        if label.dim() > 1:
            label = label.view(-1)  # Flatten if necessary

        # Calculate loss
        loss = criterion(output.unsqueeze(0), label.unsqueeze(0))  # Add batch dimension
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Make predictions
        _, predicted = torch.max(output, 0)  # Get the predicted class (0 for REAL, 1 for FAKE)

        # Update correct predictions
        correct_predictions += (predicted.item() == label.item())  # Use .item() to get the scalar value
        total_samples += 1  # Increment for each processed sample

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_graphs)
    accuracy = (correct_predictions / total_samples) * 100  # Convert to percentage

    # Append to lists
    loss_values.append(avg_loss)
    accuracy_values.append(accuracy)

    # Print epoch statistics
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Plotting the loss and accuracy
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), loss_values, label='Loss', color='blue')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), accuracy_values, label='Accuracy', color='green')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


from torch_geometric.loader import DataLoader


# In[ ]:


# Testing loop
model.eval()
test_loader = DataLoader(list(zip(test_graphs, test_labels)), batch_size=1)

correct = 0
total = 0

with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        predicted = torch.argmax(output).item()

        total += 1
        if predicted == label.item():
            correct += 1

accuracy = correct / total * 100
print(f'Test Accuracy: {accuracy:.2f}%')


# In[ ]:


# Save the model in the current directory
torch.save(model.state_dict(), 'gcn_model.pth')
print("Model saved.")


# In[ ]:


pip install networkx matplotlib


# In[ ]:


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import display, HTML

# Function to predict whether a video is real or fake
def predict_video(video_path, model, device, threshold=0.9):
    frames = extract_frames(video_path, num_frames=20)
    embeddings = extract_embeddings(frames)
    graph_data = graph_from_embeddings(embeddings, threshold=threshold).to(device)
    model.eval()
    with torch.no_grad():
        output = model(graph_data)
        predicted = torch.argmax(output).item()
    return 'REAL' if predicted == 0 else 'FAKE'

# Function to display frames from a video and calculate FPS
def display_video_frames(video_path, num_frames=5):
    capture = cv.VideoCapture(video_path)
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    # Start the timer to calculate FPS
    start_time = time.time()

    fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))
    for i, idx in enumerate(frame_indices):
        capture.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            axes[i].imshow(frame)
            axes[i].axis('off')

    # End the timer
    end_time = time.time()
    time_taken = end_time - start_time
    fps = num_frames / time_taken  # Calculate FPS based on the number of frames processed

    capture.release()
    plt.show()

    print(f"Time taken to display frames: {time_taken:.4f} seconds")
    print(f"Frame rate: {fps:.2f} frames per second (FPS)")

# Function to display video using HTML
def display_video(video_path):
    video_tag = f'''
    <video width="640" height="480" controls>
      <source src="{video_path}" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    '''
    display(HTML(video_tag))

# Function to visualize steps and predict
def visualize_and_predict(video_path, model, device, threshold=0.9):
    print(f"Processing video: {video_path}")

    # Display the video
    print("Displaying the video:")
    display_video(video_path)

    # Display sample frames from the video and calculate frame rate
    print("Displaying sample frames from the video:")
    display_video_frames(video_path)

    # Predict whether the video is real or fake
    prediction = predict_video(video_path, model, device, threshold)
    print(f"The video is predicted to be: {prediction}")

    return prediction

# Input video paths for detection
video_paths = [
    'extracted/train_sample_videos/axntxmycwd.mp4',
    'extracted/train_sample_videos/aelfnikyqj.mp4',
    'extracted/train_sample_videos/aagfhgtpmv.mp4'
]

# Process each video and calculate frame rate along with prediction
for video_path in video_paths:
    prediction = visualize_and_predict(video_path, model, device)
    print(f"Prediction for {video_path}: {prediction}")


# Vision Transformer

# In[2]:


import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

embeddings_file_path = '/content/video_embeddings.pkl'
labels_file_path = '/content/video_labels.pkl'

# Load embeddings and labels
with open(embeddings_file_path, 'rb') as f:
    embeddings = pickle.load(f)

with open(labels_file_path, 'rb') as f:
    labels = pickle.load(f)

# Convert to NumPy arrays
X = np.array(embeddings)
y = np.array(labels)

# Ensure labels are categorical
y = keras.utils.to_categorical(y, num_classes=2)

# Define the corrected Vision Transformer model
def build_vit_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Project embeddings into a sequence format
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.LayerNormalization()(x)

    # Multi-Head Self-Attention (adjust output size)
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)

    # Ensure same dimensions using Dense projection
    attn_output = layers.Dense(128)(attn_output)  # Project back to match x

    # Skip Connection
    x = layers.Add()([x, attn_output])  # Now both are (20, 128)
    x = layers.LayerNormalization()(x)

    # Feed Forward Network
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)

    # Output layer
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

# Build and compile the model
vit_model = build_vit_model(X.shape[1:], num_classes=2)
vit_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = vit_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

# Save model
vit_model.save("/content/deepfake_vit_model.h5")


# In[4]:


import matplotlib.pyplot as plt

# Plot Training and Validation Metrics
def plot_training_history(history):
    # Extract metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Accuracy Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'r*-', label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label="Training Loss")
    plt.plot(epochs, val_loss, 'r*-', label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.show()

# Call function to plot
plot_training_history(history)

