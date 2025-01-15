from PIL import Image
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pymeshlab

# Parameters
image_path = r"images\flower.png"
output_file = r"models\model.STL"

total_thickness = 2.0  # Uniform base thickness (mm)
model_thickness = total_thickness * 0.84  # Maximum additional thickness for the flower (mm)
smoothing_sigma = 1  # Sigma for Gaussian smoothing (higher = smoother)
resample_factor = 2  # Factor to resample the image for higher resolution
pixel_size = 0.1  # Pixel size in mm (resolution of the model)
side_bar = False # Add side bars to the model
side_bar_width = 2 # Width of the side bars in mm
set_dimensions = True  # Set the dimensions of the model
picture_width = 86  # Width of the picture in mm
max_height = 110  # Maximum height of the pciture in mm
flip_image = True # Flips the image horizontally

disable_picutre_view = True  # Disable the depth map view

# Step 1: Image Preparation
# Load image and normalize to [0, 1] range
image = Image.open(image_path).convert("L")
width, height = image.size

# Flip the image if flip_image is True
if flip_image:
    image = image.transpose(Image.FLIP_LEFT_RIGHT)

# Resample image based on the resample_factor (increase resolution)
if set_dimensions: # Set picture dimensions if set_dimensions is True
    # Scale image to desired width while maintaining aspect ratio
    aspect_ratio = height / width
    new_width = int(picture_width/pixel_size)
    new_height = int(new_width*aspect_ratio)

    # Resize the image to the new dimensions
    resampled_image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
    image_array = np.array(resampled_image)

    # Crop the image if the height exceeds the maximum allowed height
    max_pixel_height = int(max_height / pixel_size)
    if new_height > max_pixel_height:
        image_array = image_array[:max_pixel_height, :]
        new_height = max_pixel_height

    # Update the width and height variables
    width, height = new_width, new_height
else:
    resampled_image = image.resize((width * resample_factor, height * resample_factor), resample=Image.Resampling.BICUBIC)
    image_array = np.array(resampled_image)


# Normalize and create depth map
image_normalized = (255 - image_array) / 255  # Invert colors for depth
image_depth_map = -(image_normalized * model_thickness) + model_thickness

# Apply Gaussian smoothing
image_depth_map_smoothed = gaussian_filter(image_depth_map, sigma=smoothing_sigma)

# Visualize the depth map
if not disable_picutre_view:
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

if side_bar:
    z_wall = 0
else:
    z_wall = total_thickness

# Create engraving vertices
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
        if i == rows - 1:  # Bottom edge
            bottom_wall_vertices.append((x, y, z))
            bottom_wall_vertices.append((x, y, total_thickness))
        if j == 0:  # Left edge
            left_wall_vertices.append((x, y, z))
            left_wall_vertices.append((x, y, z_wall))
        if j == cols - 1:  # Right edge
            right_wall_vertices.append((x, y, z))
            right_wall_vertices.append((x, y, z_wall))


# Create base vertices
xmax = (cols - 1) * pixel_size
ymax = (rows - 1) * pixel_size

xmax_bar = xmax
startx = 0

if side_bar:
    xmax_bar = xmax_bar + side_bar_width
    startx = -side_bar_width

base_vertices = [(startx, 0, total_thickness), (xmax_bar, 0, total_thickness), (startx, ymax, total_thickness), (xmax_bar, ymax, total_thickness)]

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

#If side bars are disabled
# Create wall faces for the front and right walls
def create_wall_faces_fr(wall_vertices):
    wall_faces = []

    for i in range(0, len(wall_vertices) - 2, 2):
        Pn = i
        Pn1 = i + 1
        Pn2 = i + 2
        Pn3 = i + 3

        wall_faces.append([Pn, Pn3, Pn1])
        wall_faces.append([Pn, Pn2, Pn3])
    
    return wall_faces

# Create wall faces for the bottom and left walls
def create_wall_faces_bl(wall_vertices):
    wall_faces = []

    for i in range(0, len(wall_vertices) - 2, 2):
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

#Create Side Bar vertices, faces, and mesh if side bars are enabled
if side_bar:
    left_bar_vertices = [
        (startx, 0, total_thickness),  # P1 - 0
        (0, 0, total_thickness),       # P2 - 1
        (startx, 0, 0),                # P3 - 2
        (0, 0, 0),                     # P4 - 3
        (startx, ymax, total_thickness), # P1' - 4
        (0, ymax, total_thickness),    # P2' - 5
        (startx, ymax, 0),             # P3' - 6
        (0, ymax, 0)                   # P4' - 7
    ]
    right_bar_vertices = [
        (xmax, 0, total_thickness),    
        (xmax_bar, 0, total_thickness), 
        (xmax, 0, 0),                  
        (xmax_bar, 0, 0), 
        (xmax, ymax, total_thickness), 
        (xmax_bar, ymax, total_thickness), 
        (xmax, ymax, 0),              
        (xmax_bar, ymax, 0) 
    ]

    left_bar_faces = [
    [0, 2, 1], [1, 2, 3],  # Side front face
    [4, 5, 6], [5, 7, 6],  # Side back face
    [0, 4, 2], [2, 4, 6],  # Left face
    [2, 6, 7], [2, 7, 3]   # Bottom face
    ]
    right_bar_faces = [
    [0, 2, 1], [1, 2, 3],  # Side Front face
    [4, 5, 6], [5, 7, 6],  # Side back face
    [1, 7, 5], [1, 3, 7],  # Right face
    [2, 6, 7], [2, 7, 3]   # Bottom face
    ]

    left_bar_mesh = trimesh.Trimesh(vertices=left_bar_vertices, faces=left_bar_faces)
    right_bar_mesh = trimesh.Trimesh(vertices=right_bar_vertices, faces=right_bar_faces)

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
if side_bar:
    mesh = trimesh.util.concatenate([engraving_mesh, base_mesh, top_wall_mesh, bottom_wall_mesh, left_wall_mesh, right_wall_mesh, left_bar_mesh, right_bar_mesh])
else:
    mesh = trimesh.util.concatenate([engraving_mesh, base_mesh, top_wall_mesh, bottom_wall_mesh, left_wall_mesh, right_wall_mesh])

# Examine Mesh
mesh.show()

# Save the Mesh to STL file
mesh.export(output_file) 

# Step 3: Repair the Model
# Repairs the model
ms = pymeshlab.MeshSet()
ms.load_new_mesh(r'models\model.STL')

# Apply filters to repair the model
ms.apply_filter('meshing_remove_duplicate_vertices')
ms.apply_filter('meshing_remove_duplicate_faces')
ms.apply_filter('meshing_merge_close_vertices')
ms.apply_filter('meshing_snap_mismatched_borders')
ms.apply_filter('meshing_repair_non_manifold_edges')
ms.apply_filter('meshing_repair_non_manifold_vertices')
ms.save_current_mesh(r'models\model_repaired.STL')

if ymax < max_height:
    print(f"Model height: {ymax} mm")