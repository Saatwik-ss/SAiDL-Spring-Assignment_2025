import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################# Define Autoencoder Model ################################
class Encoder(nn.Module):
    def __init__(self, bottleneck_size=2):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_size)  # No activation for bottleneck
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, bottleneck_size=2):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, bottleneck_size=2):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(bottleneck_size)
        self.decoder = Decoder(bottleneck_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
###################################################################################

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=256, shuffle=False)

# Initialize Model, Loss, and Optimizer
bottleneck_size = 2
model = Autoencoder(bottleneck_size).to(device)  # Move model to GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train or Load Model
import os

if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
else:
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(device)  # Move images to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "model.pth")

model.eval()
encoded_list = []
decoded_list = []
labels = []

with torch.no_grad():
    for images, y in test_loader:
        images = images.to(device) 
        encoded = model.encoder(images).cpu().numpy()  # Convert to NumPy for visualization
        decoded = model.decoder(torch.tensor(encoded, dtype=torch.float32).to(device)).cpu().numpy()
        encoded_list.append(encoded)
        decoded_list.append(decoded)
        labels.extend(y.numpy())

encoded_imgs = np.vstack(encoded_list)
decoded_imgs = np.vstack(decoded_list)
labels = np.array(labels)

# Generate Mesh Grid for Latent Space
mesh = {}
x_range = list(range(-90, 90, 5))
y_range = list(range(-90, 90, 5))

for x in tqdm(x_range):
    for y in y_range:
        if x not in mesh:
            mesh[x] = {}
        latent_vector = torch.tensor([[x, y]], dtype=torch.float32, device=device) 
        with torch.no_grad():
            mesh[x][y] = model.decoder(latent_vector).cpu().numpy().reshape(28, 28)  

arr = np.zeros((180, 180, 28, 28))
for x in x_range:
    for y in y_range:
        arr[(x + 90) // 5][(y + 90) // 5] = mesh[x][y] 

del mesh  # Free memory

# Plot Encoded Data and Interactive Reconstruction
fig, ax = plt.subplots(1, 2)
ax[0].scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=labels, s=8, cmap='tab10')


def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    ix = int((event.xdata + 90) / 5)
    iy = int((event.ydata + 90) / 5)
    if 0 <= ix < arr.shape[0] and 0 <= iy < arr.shape[1]:
        ax[1].imshow(arr[ix][iy], cmap='gray')
        plt.draw()


fig.canvas.mpl_connect('motion_notify_event', onclick)
plt.show()
