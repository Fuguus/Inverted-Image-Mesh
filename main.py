from PIL import Image
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Parameters
image_path = r"images\flower.png"
output_file = r"models\model.STL"
total_thickness = 1.0  # Uniform base thickness (mm)
model_thickness = total_thickness * 0.9  # Maximum additional thickness for the flower (mm)
smoothing_sigma = 1  # Sigma for Gaussian smoothing (higher = smoother)
resample_factor = 2  # Factor to resample the image for higher resolution
pixel_size = 0.1  # Pixel size in mm (resolution of the model)

# Step 1: Image Preparation
# Load image and normalize to [0, 1] range
image = Image.open(image_path).convert("L")
width, height = image.size

# Resample image based on the resample_factor (increase resolution)
resampled_image = image.resize((width * resample_factor, height * resample_factor), resample=Image.Resampling.BICUBIC)
image_array = np.array(resampled_image)

# Normalize and create depth map
image_normalized = (255 - image_array) / 255  # Invert colors for depth
image_depth_map = -(image_normalized * model_thickness) + model_thickness

# Apply Gaussian smoothing
image_depth_map_smoothed = gaussian_filter(image_depth_map, sigma=smoothing_sigma)

# Visualize the depth map
plt.imshow(image_depth_map_smoothed, cmap="gray")
plt.colorbar(label="Thickness (mm)")
plt.title("Smoothed Heightmap with Flat Base")
plt.show()

# Step 2: Create 3D Model

# Create image mesh
rows, cols = image_depth_map_smoothed.shape

# Initialize lists to store vertices
engraving_vertices = []
top_wall_vertices = []
bottom_wall_vertices = []
left_wall_vertices = []
right_wall_vertices = []
base_vertices = []

# Define engraving and wall vertices with scaling based on pixel_size
for i in range(rows):
    for j in range(cols):
        # Convert pixel positions to physical space (mm) by scaling by pixel_size
        x = j * pixel_size  # Convert to mm
        y = i * pixel_size  # Convert to mm
        z = image_depth_map_smoothed[i, j]  # Depth from the depth map (mm)
        engraving_vertices.append((x, y, z))  # Append the vertex with scaled (x, y) and depth (z)
        if i == 0:  # Top edge
            top_wall_vertices.append((x, y, z))
            top_wall_vertices.append((x, y, total_thickness))
        elif i == rows - 1:  # Bottom edge
            bottom_wall_vertices.append((x, y, z))
            bottom_wall_vertices.append((x, y, total_thickness))
        elif j == 0:  # Left edge
            left_wall_vertices.append((x, y, z))
            left_wall_vertices.append((x, y, total_thickness))
        elif j == cols - 1:  # Right edge
            right_wall_vertices.append((x, y, z))
            right_wall_vertices.append((x, y, total_thickness))

# Create base vertices
xmax = cols * pixel_size
ymax = rows * pixel_size

base_vertices = [(0, 0, total_thickness), (xmax, 0, total_thickness), (0, ymax, total_thickness), (xmax, ymax, total_thickness)]

# Initialize lists to store faces
engraving_faces = []
wall_faces = []
base_faces = []

# Create engraving faces
for i in range(rows - 1):
    for j in range(cols - 1):
        # Create two triangular faces for each grid cell
        engraving_faces.append([(i + 1) * cols + j, i * cols + j + 1, i * cols + j])
        engraving_faces.append([(i + 1) * cols + j + 1, i * cols + j + 1, (i + 1) * cols + j])

# Create wall faces for the front and back walls
def create_wall_faces_fr(wall_vertices):
    wall_faces = []

    for i in range(0, len(wall_vertices) -2, 2):
        Pn = i
        Pn1 = i + 1
        Pn2 = i + 2
        Pn3 = i + 3

        wall_faces.append([Pn, Pn3, Pn1])
        wall_faces.append([Pn, Pn2, Pn3])
    
    return wall_faces

# Create wall faces for the front and back walls
def create_wall_faces_bl(wall_vertices):
    wall_faces = []

    for i in range(0, len(wall_vertices) -2, 2):
        Pn = i
        Pn1 = i + 1
        Pn2 = i + 2
        Pn3 = i + 3

        wall_faces.append([Pn3, Pn, Pn1])
        wall_faces.append([Pn3, Pn2, Pn])
    
    return wall_faces

top_wall_faces = create_wall_faces_fr(top_wall_vertices)
bottom_wall_faces = create_wall_faces_bl(bottom_wall_vertices)
left_wall_faces = create_wall_faces_bl(left_wall_vertices)
right_wall_faces = create_wall_faces_fr(right_wall_vertices)

wall_faces = top_wall_faces + bottom_wall_faces + left_wall_faces + right_wall_faces

# Create final flat base mesh
base_faces = []
base_faces.append([0, 1, 2])
base_faces.append([1, 3, 2])

# Create the mesh
engraving_mesh = trimesh.Trimesh(vertices=engraving_vertices, faces=engraving_faces)
top_wall_mesh = trimesh.Trimesh(vertices=top_wall_vertices, faces=top_wall_faces)
bottom_wall_mesh = trimesh.Trimesh(vertices=bottom_wall_vertices, faces=bottom_wall_faces)
left_wall_mesh = trimesh.Trimesh(vertices=left_wall_vertices, faces=left_wall_faces)
right_wall_mesh = trimesh.Trimesh(vertices=right_wall_vertices, faces=right_wall_faces)
base_mesh = trimesh.Trimesh(vertices=base_vertices, faces=base_faces)

# Combine mesh
mesh = trimesh.util.concatenate([engraving_mesh, base_mesh, top_wall_mesh, bottom_wall_mesh, left_wall_mesh, right_wall_mesh])

# Examine Mesh
mesh.show()

# Save the Mesh to STL file
mesh.export(output_file) 