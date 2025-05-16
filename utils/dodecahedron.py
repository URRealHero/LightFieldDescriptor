import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from typing import List
import matplotlib.pyplot as plt

class DodecahedronCameraSystem:
    def __init__(self, base_directions: np.ndarray = None, rotations: List[np.ndarray] = None):
        """
        Initialize the Dodecahedron Camera System.

        Parameters
        ----------
        base_directions : np.ndarray
            Array of shape (20, 3) representing the base camera directions (unit vectors).
        rotations : list of np.ndarray
            List of 3x3 rotation matrices representing global orientations.
        """
        self.base_directions = base_directions if base_directions is not None else self._generate_base_directions()
        self.rotations = rotations if rotations is not None else self._generate_global_rotations()

    def _generate_base_directions(self) -> np.ndarray:
        """Generate the 20 unit directions from the vertices of a regular dodecahedron."""
        phi = (1 + np.sqrt(5)) / 2
        verts = []

        coords = [-1, 1]
        for x in coords:
            for y in coords:
                for z in coords:
                    verts.append([x, y, z])

        for x in coords:
            for y in coords:
                verts.append([0, x / phi, y * phi])
                verts.append([x / phi, y * phi, 0])
                verts.append([x * phi, 0, y / phi])

        verts = np.array(verts, dtype=np.float32)
        return verts / np.linalg.norm(verts, axis=1, keepdims=True)

    def _fibonacci_sphere(self, samples=10):
        """
        Generate approximately uniformly distributed points on a sphere.
        xz -> horizontal plane, y -> vertical axis.

        The i-th point is computed using:

            y_i   = 1 - 2*(i + 0.5)/N
            r_i   = sqrt(1 - y_i^2)
            phi_i = i * golden_angle
            x_i   = r_i * cos(phi_i)
            z_i   = r_i * sin(phi_i)

        Where:
            - golden_angle = π * (3 - sqrt(5)) ≈ 2.39996 radians
            - i ∈ {0, 1, ..., N - 1}

        Returns
        -------
        points : np.ndarray
            A (samples, 3) array of unit vectors on the sphere surface.
        """

        
        points = []
        offset = 2.0 / samples # 2/N -> used for offset*n
        increment = np.pi * (3.0 - np.sqrt(5.0)) # This is the constant: golden angle, its formula in docstring
        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - y * y)
            phi = i * increment
            x = np.cos(phi) * r
            z = np.sin(phi) * r
            points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    def _look_at_rotation(self, forward: np.ndarray, up=np.array([0, 1, 0])) -> np.ndarray:
        """Create a rotation matrix that looks in the 'forward' direction."""
        z = forward / np.linalg.norm(forward)
        if np.abs(np.dot(z, up)) > 0.99:
            up = np.array([1, 0, 0])
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=1)

    def _generate_global_rotations(self, num=10) -> List:
        """Generate 10 global rotation matrices from Fibonacci sphere points."""
        directions = self._fibonacci_sphere(num)
        return [self._look_at_rotation(d) for d in directions]

    def get_camera_system(self, index: int) -> np.ndarray:
        """
        Apply the index-th global rotation to the base dodecahedron directions.

        Parameters
        ----------
        index : int
            Index of the rotation to apply.

        Returns
        -------
        np.ndarray
            Rotated camera directions (20, 3).
        """
        return np.dot(self.base_directions, self.rotations[index].T)

    def save(self, path: str):
        """Save the camera system to a .npz file."""
        np.savez(path, base=self.base_directions, rotations=np.array(self.rotations))

    @staticmethod
    def load(path: str):
        """Load a camera system from a .npz file."""
        data = np.load(path, allow_pickle=True)
        base = data['base']
        rotations = list(data['rotations'])
        return DodecahedronCameraSystem(base_directions=base, rotations=rotations)


# Create and save the camera system
camera_system = DodecahedronCameraSystem()
os.makedirs("camera_system", exist_ok=True)
camera_system.save("camera_system/dodeca_camera_system.npz")

# ------------------Visualization----------------------------#
def visualize_camera_directions(directions: np.ndarray, title="Dodecahedron Camera Directions"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw unit sphere for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.1)

    # Plot camera directions
    for d in directions:
        ax.quiver(0, 0, 0, d[0], d[1], d[2], color='red', arrow_length_ratio=0.1)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()
    
    
camera_system = DodecahedronCameraSystem.load("camera_system/dodeca_camera_system.npz")
directions = camera_system.get_camera_system(2)
visualize_camera_directions(directions, title="Rotated Dodecahedron Camera System #0")