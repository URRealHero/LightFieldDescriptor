import open3d as o3d
from typing_extensions import Optional, Union

def load_obj(filename: str) -> o3d.geometry.TriangleMesh:
    try:
        mesh = o3d.io.read_triangle_mesh(filename)
        return mesh
    except Exception as e:
        print(f"Error loading mesh from {filename}: {e}")
        return None

def normalize_mesh(mesh: Union[o3d.geometry.TriangleMesh, str]) -> o3d.geometry.TriangleMesh:
    # Compute the center of the mesh
    if isinstance(mesh, str):
        mesh = load_obj(mesh)
        if mesh is None:
            raise ValueError("Could not load mesh from the provided filename.")
    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    scale = 1.0 / max(bbox.get_extent())  # scale into unit cube
    mesh.translate(-center)
    mesh.scale(scale, center=(0, 0, 0))
    return mesh

def save_mesh(mesh: o3d.geometry.TriangleMesh, filename: str):
    """
    Save an Open3D TriangleMesh to a file.

    Parameters:
    ----------
    mesh : o3d.geometry.TriangleMesh
        The mesh to save.
    filename : str
        Output file path (.ply or .obj).
    """
    if not o3d.io.write_triangle_mesh(filename, mesh):
        raise IOError(f"Failed to save mesh to {filename}")
