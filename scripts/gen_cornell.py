# Cornell Box generator
def gen_cornell():
    obj = []
    v = []
    # Vertices (x, y, z)
    # Floor
    v += [(-5,0,-5), (5,0,-5), (5,0,5), (-5,0,5)] # 1-4
    # Ceiling
    v += [(-5,10,-5), (5,10,-5), (5,10,5), (-5,10,5)] # 5-8
    # Back wall
    v += [(-5,0,-5), (5,0,-5), (5,10,-5), (-5,10,-5)] # 9-12
    # Left wall
    v += [(-5,0,-5), (-5,10,-5), (-5,10,5), (-5,0,5)] # 13-16
    # Right wall
    v += [(5,0,-5), (5,10,-5), (5,10,5), (5,0,5)] # 17-20
    # Short box
    v += [(-3,0,1), (0,0,1), (0,0,4), (-3,0,4), (-3,3,1), (0,3,1), (0,3,4), (-3,3,4)] # 21-28
    # Tall box
    v += [(1,0,-3), (4,0,-3), (4,0,0), (1,0,0), (1,6,-3), (4,6,-3), (4,6,0), (1,6,0)] # 29-36
    # Light
    v += [(-2,9.9,-2), (2,9.9,-2), (2,9.9,2), (-2,9.9,2)] # 37-40

    for x, y, z in v:
        obj.append(f"v {x} {y} {z}")

    # Faces (1-based)
    def face(indices):
        obj.append(f"f {' '.join(map(str, indices))}")

    face([4,3,2,1]) # Floor -> UP
    face([5,6,7,8]) # Ceiling -> DOWN
    face([9,10,11,12]) # Back -> +Z
    face([13,14,15,16]) # Left -> +X
    face([17,20,19,18]) # Right -> -X
    
    # Short box (outward facing)
    face([21,22,23,24]); face([28,27,26,25]) # bot/top
    face([21,25,26,22]); face([22,26,27,23]) # front/right
    face([23,27,28,24]); face([24,28,25,21]) # back/left

    # Tall box (outward facing)
    face([29,30,31,32]); face([36,35,34,33]) # bot/top
    face([29,33,34,30]); face([30,34,35,31]) # front/right
    face([31,35,36,32]); face([32,36,33,29]) # back/left

    face([37,38,39,40]) # Light -> DOWN

    with open("/home/ethan/dev/learning/xenon/scenes/meshes/cornell_box.obj", "w") as f:
        f.write("\n".join(obj))

gen_cornell()
