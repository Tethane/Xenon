import os

def gen_cow_scene():
    return
    scene_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(scene_dir, "cow.xenon")
    
    with open(output_path, "w") as f:
        f.write("config 800 600 256\n")
        f.write("camera 0 5 15 0 5 0 40\n")
        
        # material name r g b metallic roughness spec ior trans trans_rough
        f.write("material floor 0.725 0.71 0.68 0 0.5 0.5 1.5 0 -1\n")
        f.write("material ceiling 0.725 0.71 0.68 0 0.5 0.5 1.5 0 -1\n")
        f.write("material back 0.725 0.71 0.68 0 0.5 0.5 1.5 0 -1\n")
        f.write("material left 0.63 0.065 0.05 0 0.5 0.5 1.5 0 -1\n")
        f.write("material right 0.14 0.45 0.091 0 0.5 0.5 1.5 0 -1\n")
        f.write("material white 0.725 0.71 0.68 0 0.5 0.5 1.5 0 -1\n")
        f.write("material glossy_metal 0.9 0.9 0.9 1.0 0.1 1.0 1.5 0 -1\n")
        f.write("material light 15 15 15 0 0 0 1.5 0 -1\n")
        
        # Meshes: path px py pz scale rx ry rz
        # Empty Box
        f.write("mesh scenes/cornell_box/meshes/empty_box.obj 0 0 0 1 0 0 0\n")
        
        # Cow: Centered, scaled to fit.
        f.write("mesh scenes/cow/meshes/cow.obj 0 3.8 0 1.1 0 -45 0\n")
        
        # Lights (Indices in empty_box.obj)
        f.write("light 0 10 15 15 15 1\n")
        f.write("light 0 11 15 15 15 1\n")

if __name__ == "__main__":
    gen_cow_scene()
    print("Generated cow.xenon")
