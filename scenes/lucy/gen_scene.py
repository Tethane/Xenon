import os

def gen_lucy_scene():
    scene_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(scene_dir, "lucy.xenon")
    
    with open(output_path, "w") as f:
        f.write("config 800 600 256\n")
        f.write("camera 0 5 15 0 5 0 40\n")
        
        # Materials
        f.write("material floor 0.725 0.71 0.68 0 0.5\n")
        f.write("material ceiling 0.725 0.71 0.68 0 0.5\n")
        f.write("material back 0.725 0.71 0.68 0 0.5\n")
        f.write("material left 0.63 0.065 0.05 0 0.5\n")
        f.write("material right 0.14 0.45 0.091 0 0.5\n")
        f.write("material glass 1.0 1.0 1.0 0.0 0.01\n") 
        f.write("material light 15 15 15 0 0\n")
        
        # Meshes: path px py pz scale rx ry rz
        f.write("mesh scenes/cornell_box/meshes/empty_box.obj 0 0 0 1 0 0 0\n")
        
        # Lucy: She is massive and far from origin.
        # We need to center her first. Let's assume her midpoint is around (600, -120, 240).
        # We can do this by using a transform that translates her back to origin, scales, then translates to box center.
        # However, our transform is Scale -> Rot -> Translate.
        # So we better hope she's somewhat centered or we scale her down enough.
        # Let's try scale 0.008 and offset.
        f.write("mesh scenes/lucy/meshes/lucy.obj -4.8 1.0 -2.0 0.008 0 180 0\n")
        
        # Lights
        f.write("light 10 15 15 15 1\n")
        f.write("light 11 15 15 15 1\n")

if __name__ == "__main__":
    gen_lucy_scene()
    print("Generated lucy.xenon")
