import os

def gen_lucy_scene():
    return
    scene_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(scene_dir, "lucy.xenon")
    
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
        
        # Glass Lucy
        f.write("material glass 1.0 1.0 1.0 0.0 0.00 1.0 1.5 1.0 0.0\n") 
        f.write("material light 15 15 15 0 0 0 1.5 0 -1\n")
        
        # Meshes: path px py pz scale rx ry rz
        f.write("mesh scenes/cornell_box/meshes/empty_box.obj 0 0 0 1 0 0 0\n")
        
        # Lucy: Centered and sitting on floor
        # Scaled bounds were roughly [-3, 4] in X, [-3.1, 1.1] in Y, [-4.8, 7.9] in Z.
        # We need to shift her to be centered in X and Z, and floor-aligned in Y.
        f.write("mesh scenes/lucy/meshes/lucy.obj 3 3 0 0.0065 90 180 0\n")
        
        # Lights
        f.write("light 0 10 15 15 15 1\n")
        f.write("light 0 11 15 15 15 1\n")

if __name__ == "__main__":
    gen_lucy_scene()
    print("Generated lucy.xenon")
