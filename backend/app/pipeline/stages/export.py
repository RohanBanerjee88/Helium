"""
Stage: export (Mesh Extraction + STL/OBJ Export)

TODO: This is a placeholder. Implement one of the following:

Option A — Open3D Poisson surface reconstruction:
    pcd = o3d.io.read_point_cloud(str(dense_ply))
    pcd.estimate_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    o3d.io.write_triangle_mesh(str(mesh_dir / "mesh.obj"), mesh)

Option B — trimesh for repair + export:
    import trimesh
    mesh = trimesh.load(str(mesh_obj))
    mesh.fill_holes()
    mesh.export(str(mesh_dir / "mesh.stl"))

Expected inputs:
    artifacts/point_cloud/dense.ply

Expected outputs:
    artifacts/mesh/mesh.obj
    artifacts/mesh/mesh.stl
"""

import json
from pathlib import Path
from typing import Any, Dict


def run(artifacts_dir: Path) -> Dict[str, Any]:
    mesh_dir = artifacts_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "status": "placeholder",
        "note": (
            "Mesh export not yet implemented. "
            "Connect Open3D Poisson reconstruction + trimesh export here."
        ),
        "expected_inputs": ["dense.ply"],
        "expected_outputs": ["mesh.obj", "mesh.stl"],
    }

    with open(mesh_dir / "export_status.json", "w") as fh:
        json.dump(status, fh, indent=2)

    return status
