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
        f.write("material white 0.725 0.71 0.68 0 0.5\n")
        
        # Dielectric Lucy (Glass)
        # Assuming PrincipledBSDF supports transmission as implemented earlier
        # material name r g b metallic roughness ior specular transmission
        # Our current material line parser: ss >> name >> m.albedo.x >> m.albedo.y >> m.albedo.z >> m.metallic >> m.roughness;
        # Wait, let me check the loader again. It only parses 5 floats.
        f.write("material glass 1.0 1.0 1.0 0.0 0.01\n") 
        f.write("material light 15 15 15 0 0\n")
        
        # Meshes
        f.write("mesh scenes/cornell_box/meshes/cornell_box.obj\n")
        f.write("mesh scenes/lucy/meshes/lucy.obj\n")
        
        # Lights (using indices from cornell_box.obj)
        f.write("light 34 15 15 15 1\n")
        f.write("light 35 15 15 15 1\n")

if __name__ == "__main__":
    gen_lucy_scene()
    print("Generated lucy.xenon")
