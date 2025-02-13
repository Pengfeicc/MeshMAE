import open3d as o3d
import numpy as np
from pathlib import Path
import json
import sys

color_map = {
    1: [1, 0, 0],  # Red
    2: [0, 1, 0],  # Green
    3: [0, 0, 1],  # Blue
    4: [1, 1, 0],  # Yellow
}

fname = "example"

# filenames
fpreds = Path("predictions/19-009.txt")
fobj = Path("dataset/xxx xxx/test/alien/19-009.obj")
fjson = Path("dataset/xxx xxx/test/alien/19-009.json")

# load files
mesh = o3d.io.read_triangle_mesh(str(fobj)).compute_vertex_normals()
preds = np.loadtxt(fpreds).astype(int) # 将fpreds转换成numpy格式, 输出[2,2,2 ... 0,0,0]共计16384个vertices


with open(fjson, "r") as f: # “r”表示只读取文件内容
    data = json.load(f)
    raw_labels = data["raw_labels"] # 从 data 字典中提取名为 "raw_labels" 对应的值，并将其赋给 raw_labels
    sub_labels = data["sub_labels"] # # 从 data 字典中提取名为 "sub_labels" 对应的值，并将其赋给 sub_labels

    print(type(raw_labels), len(raw_labels))
    print(type(sub_labels), len(sub_labels))

preds = preds.tolist()

vertex_colors = np.zeros_like(mesh.vertices) # 初始化，其形状（shape）与 mesh.vertices 相同

for face_id in range(len(mesh.triangles)): # 遍历所有三角网格的数量
    label = preds[face_id]
    color = color_map.get(label, [1, 1, 1])  # Default to white if label not found
    vertex_colors[mesh.triangles[face_id]] = color
mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
o3d.visualization.draw_geometries([mesh])
