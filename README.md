# LightFieldDescriptor
An unofficial implementation of paper "On Visual Similarity Based 3D Model Retrieval"

```
.
├── render.py                  # 🔄 Main script to batch-render silhouettes for the dataset
├── lfd.py                     # 🧩 Full LFD pipeline using image metrics (step 4, to be implemented)

├── utils/                     # 🔧 Utility modules
│   ├── dodecahedron.py        # 📐 Defines the camera system and its 10 global orientations
│   ├── normalize.py           # 📏 Mesh normalization into unit cube
│   ├── render_lfd.py          # 📷 Renderer class for generating silhouette images
│   └── image_metric.py        # 📊 Computes descriptor distance (step 4, to be implemented)

├── data/                      # 📦 Data interface and loaders
│   └── ShapenetLoader.py      # 📂 PyTorch-style DataLoader for ShapeNet mesh paths

├── camera_system/             # 💾 Precomputed camera directions and rotations
│   └── dodeca_camera_system.npz

└── README.md                  
```