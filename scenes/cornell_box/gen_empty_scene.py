import os

def gen_empty_scene():
    scene_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(scene_dir, "cornell_empty.xenon")
    
    with open(output_path, "w") as f:
        f.write("config 800 600 256\n")
        f.write("camera 0 5 15 0 5 0 40\n")
        
        # Materials
        f.write("material floor 0.725 0.71 0.68 0 0.5\n")
        f.write("material ceiling 0.725 0.71 0.68 0 0.5\n")
        f.write("material back 0.725 0.71 0.68 0 0.5\n")
        f.write("material left 0.63 0.065 0.05 0 0.5\n")
        f.write("material right 0.14 0.45 0.091 0 0.5\n")
        f.write("material white 0.725 0.71 0.68 0 0.5\n")
        f.write("material light 15 15 15 0 0\n")
        
        # Mesh
        f.write("mesh scenes/cornell_box/meshes/empty_box.obj 0 0 0 1 0 0 0\n")
        
        # Lights
        f.write("light 0 10 15 15 15 1\n")
        f.write("light 0 11 15 15 15 1\n")

if __name__ == "__main__":
    gen_empty_scene()
    print("Generated cornell_empty.xenon")
