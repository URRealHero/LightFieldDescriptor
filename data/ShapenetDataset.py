import os
import json
from torch.utils.data import Dataset

class ShapenetMeshDataset(Dataset):
    def __init__(self, metadata_path):
        """
        Parameters
        ----------
        metadata_path : str
            Path to a JSON file containing a list of .obj file paths.
        """
        with open(metadata_path, 'r') as f:
            self.mesh_paths = json.load(f)

        # Optional sanity check
        self.mesh_paths = [p for p in self.mesh_paths if os.path.exists(p)]

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        mesh_path = self.mesh_paths[idx]

        # Define output directory: same as mesh path's parent, but with LFD_render/
        base_dir = os.path.dirname(mesh_path)
        output_dir = os.path.join(base_dir, "LFD_render")

        return {
            "mesh_path": mesh_path,
            "output_dir": output_dir
        }


