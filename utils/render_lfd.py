from .dodecahedron import *
from .normalize import *
import open3d as o3d
import numpy as np

class LightFieldRenderer:
    def __init__(self, camera_system_path=None, image_size=256):
        # Load DodecahedronCameraSystem
        if camera_system_path is None:
            self.camera_system = DodecahedronCameraSystem()
        else:
            self.camera_system = DodecahedronCameraSystem.load(camera_system_path)
        self.image_size = image_size

    def render_lfd(self, mesh_path, out_dir):
        # Load and normalize mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        mesh = normalize_mesh(mesh)

        for lfd_id in range(10):  # 10 global orientations
            directions = self.camera_system.get_camera_system(lfd_id)[:10]  # select 10 unique views for this LFD
            for view_id, dir_vec in enumerate(directions):
                binary_img = self._render_view(mesh, dir_vec)
                self._save_image(binary_img, out_dir, lfd_id, view_id)

    def _render_view(self, mesh, direction):
        # Compute eye, lookat, up → create orthographic camera
        # Use OffscreenRenderer to render binary silhouette
        # Return image as numpy array (uint8, shape HxW)
        eye, lookat, up = get_camera_pose(direction)
        
        
        # setup a renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(self.image_size, self.image_size)
        # white bg
        renderer.scene.set_background([1.0,1.0,1.0,1.0])
        
        # Set up material (unlit black surface)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [0.0, 0.0, 0.0, 1.0]  # black
        
        
        renderer.scene.add_geometry("mesh", mesh, mat)
        
        # Set orthographic projection
        renderer.scene.camera.set_projection(
            projection_type=o3d.visualization.rendering.Camera.Projection.ORTHOGRAPHIC,
            left=-1.2, right=1.2, bottom=-1.2, top=1.2, near=0.1, far=10.0
        )
        
        renderer.scene.camera.look_at(lookat, eye, up)
        
        img = renderer.render_to_image()
        
        gray = np.asarray(img)[:, :, 0]  # extract grayscale
        
        # Binarize
        binary = (gray < 240).astype(np.uint8) * 255
            
        renderer.scene.clear_geometry()
        return binary


    def _save_image(self, img, out_dir, lfd_id, view_id):
        from PIL import Image
        import os

        # Create output folder if needed
        os.makedirs(out_dir, exist_ok=True)

        # Define filename
        filename = f"LFD{lfd_id}_view{view_id}.png"
        save_path = os.path.join(out_dir, filename)

        # Save binary image
        Image.fromarray(img).save(save_path)

def get_camera_pose(direction, distance=2.5):
    eye = direction * distance
    lookat = np.array([0, 0, 0])
    up = np.array([0, 1, 0]) if abs(np.dot(direction, [0, 1, 0])) < 0.99 else np.array([1, 0, 0])
    return eye, lookat, up