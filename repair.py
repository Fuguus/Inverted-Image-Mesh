import pymeshlab

# Repairs the model
ms = pymeshlab.MeshSet()
ms.load_new_mesh(r'models\model.STL')

ms.apply_filter('meshing_remove_duplicate_vertices')
ms.apply_filter('meshing_remove_duplicate_faces')
ms.apply_filter('meshing_merge_close_vertices')
ms.apply_filter('meshing_snap_mismatched_borders')
ms.apply_filter('meshing_repair_non_manifold_edges')
ms.apply_filter('meshing_repair_non_manifold_vertices')
ms.save_current_mesh(r'models\model_repaired.STL')