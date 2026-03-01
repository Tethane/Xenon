import os

def gen_cow_scene():
    scene_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(scene_dir, "cow.xenon")
    
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
        f.write("material glossy_metal 0.1 0.1 0.1 0.9 0.1\n") # Metallic cow
        f.write("material light 15 15 15 0 0\n")
        
        # Meshes
        # We assume cornell_box.obj is available in cow/meshes for assembly if needed, 
        # but for now let's just use the cow and the box from the other folder to test merging.
        f.write("mesh scenes/cornell_box/meshes/cornell_box.obj\n")
        f.write("mesh scenes/cow/meshes/cow.obj\n")
        
        # Lights (using indices from cornell_box.obj)
        f.write("light 34 15 15 15 1\n")
        f.write("light 35 15 15 15 1\n")

if __name__ == "__main__":
    gen_cow_scene()
    print("Generated cow.xenon")
