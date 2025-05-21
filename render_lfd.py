from torch.utils.data import DataLoader
from data.ShapenetDataset import ShapenetMeshDataset
from utils.render_lfd import LightFieldRenderer

dataset = ShapenetMeshDataset("./Shapenet/dataset/shapenet_obj.json")
loader = DataLoader(dataset, batch_size=1, shuffle=False)
renderer = LightFieldRenderer(image_size=256)


for batch in loader:
    mesh_path = batch["mesh_path"][0]
    output_dir = batch["output_dir"][0]
    
    renderer.render_lfd(mesh_path, output_dir)