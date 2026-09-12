"""Rig the static jacket mesh with a simple humanoid skeleton and write a skinned GLB."""
import numpy as np, struct, json, sys
from pygltflib import GLTF2, Node, Skin, Mesh, Primitive, Attributes, Accessor, BufferView, Buffer, Material, PbrMetallicRoughness, Scene

SRC, MESH_NAME, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
g = GLTF2().load(SRC)
blob = g.binary_blob()

def read(acc_idx):
    a = g.accessors[acc_idx]; bv = g.bufferViews[a.bufferView]
    ncomp = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}[a.type]
    dt = {5126:np.float32,5123:np.uint16,5125:np.uint32,5121:np.uint8}[a.componentType]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    return np.frombuffer(blob, dtype=dt, count=a.count*ncomp, offset=off).reshape(a.count, ncomp).copy()

mi = [i for i,m in enumerate(g.meshes) if m.name==MESH_NAME][0]
node = [n for n in g.nodes if n.mesh==mi][0]
prim = g.meshes[mi].primitives[0]
P = read(prim.attributes.POSITION).astype(np.float64)
N = read(prim.attributes.NORMAL).astype(np.float64)
UV = read(prim.attributes.TEXCOORD_0)
I = read(prim.indices).reshape(-1).astype(np.uint32)

# node transform (rotation + translation)
M = np.eye(4)
if node.matrix: M = np.array(node.matrix).reshape(4,4).T
else:
    if node.rotation:
        x,y,z,w = node.rotation
        M[:3,:3] = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    if node.translation: M[:3,3] = node.translation
P = P @ M[:3,:3].T + M[:3,3]; N = N @ M[:3,:3].T
# recenter: x,z center, y bottom at 0
lo, hi = P.min(0), P.max(0)
P -= [ (lo[0]+hi[0])/2, lo[1], (lo[2]+hi[2])/2 ]
lo, hi = P.min(0), P.max(0); H = hi[1]; W = hi[0]-lo[0]
print("centered bounds", lo.round(3), hi.round(3))

# --- detect landmarks from geometry ---
def centroid(mask): return P[mask].mean(0)
cuffL = centroid(P[:,0] < lo[0]+0.06); cuffR = centroid(P[:,0] > hi[0]-0.06)
top = P[P[:,1] > H-0.10]
shL = np.array([top[:,0].min()+0.05, H-0.07, top[top[:,0]<top[:,0].min()+0.08][:,2].mean()])
shR = np.array([top[:,0].max()-0.05, H-0.07, top[top[:,0]>top[:,0].max()-0.08][:,2].mean()])
elL = shL + 0.5*(cuffL-shL); elR = shR + 0.5*(cuffR-shR)
hips = np.array([0,0.02,0.0]); spine = np.array([0,0.5*H,0.0]); neck = np.array([0,H-0.03,0.0])
joints = {"hips":hips,"spine":spine,"neck":neck,
          "upperarm_l":shL,"lowerarm_l":elL,"hand_l":cuffL,
          "upperarm_r":shR,"lowerarm_r":elR,"hand_r":cuffR}
parent = {"hips":None,"spine":"hips","neck":"spine","upperarm_l":"spine","lowerarm_l":"upperarm_l","hand_l":"lowerarm_l",
          "upperarm_r":"spine","lowerarm_r":"upperarm_r","hand_r":"lowerarm_r"}
names = list(joints.keys()); jid = {n:i for i,n in enumerate(names)}
for n in names: print(f"  {n:12s}", joints[n].round(3))

# --- skin weights ---
def seg_dist(p, a, b):
    ab = b-a; t = np.clip(((p-a)@ab)/(ab@ab), 0, 1); return np.linalg.norm(p-(a+np.outer(t,ab)),axis=1), t
def smooth(t): t=np.clip(t,0,1); return t*t*(3-2*t)
W4 = np.zeros((len(P),4),np.float32); J4 = np.zeros((len(P),4),np.uint8)
dL,tL = seg_dist(P, shL, cuffL); dR,tR = seg_dist(P, shR, cuffR)
torso_half = 0.22*W/0.88
n_t = np.abs(P[:,0])/torso_half
n_L = dL/0.10; n_R = dR/0.10
for i in range(len(P)):
    w = {}
    # torso vs arm membership
    aL = smooth((n_t[i]-n_L[i]+0.5)/1.0) if P[i,0]<0 else 0.0
    aR = smooth((n_t[i]-n_R[i]+0.5)/1.0) if P[i,0]>0 else 0.0
    arm = max(aL,aR); tor = 1-arm
    y = P[i,1]
    if tor>0:
        if y < 0.5*H:
            k = smooth(y/(0.5*H)); w["hips"]=tor*(1-k)*0.6; w["spine"]=tor*(k+ (1-k)*0.4)
        else:
            k = smooth((y-0.5*H)/(0.5*H)); w["spine"]=tor*(1-k*0.5); w["neck"]=tor*k*0.5
    if aL>0:
        k = smooth((tL[i]-0.42)/0.16); w["upperarm_l"]=aL*(1-k); w["lowerarm_l"]=aL*k
    if aR>0:
        k = smooth((tR[i]-0.42)/0.16); w["upperarm_r"]=aR*(1-k); w["lowerarm_r"]=aR*k
    items = sorted(w.items(), key=lambda kv:-kv[1])[:4]
    s = sum(v for _,v in items)
    for c,(n,v) in enumerate(items):
        if v>0: J4[i,c]=jid[n]; W4[i,c]=v/s
print("weight sums ok:", np.allclose(W4.sum(1),1,atol=1e-4))

# --- build glTF ---
out = GLTF2(); out.asset.version="2.0"
bufs = []
def add(arr, target=None, comp=None, typ=None, minmax=False):
    arr = np.ascontiguousarray(arr); data = arr.tobytes()
    off = sum(len(b) for b in bufs); pad = (4-len(data)%4)%4; bufs.append(data+b"\0"*pad)
    out.bufferViews.append(BufferView(buffer=0, byteOffset=off, byteLength=len(data), target=target))
    acc = Accessor(bufferView=len(out.bufferViews)-1, componentType=comp, count=arr.shape[0] if arr.ndim>1 else len(arr), type=typ)
    if minmax: acc.min=arr.min(0).tolist(); acc.max=arr.max(0).tolist()
    out.accessors.append(acc); return len(out.accessors)-1
ACC_POS = add(P.astype(np.float32),34962,5126,"VEC3",True)
ACC_NRM = add(N.astype(np.float32),34962,5126,"VEC3")
ACC_UV  = add(UV.astype(np.float32),34962,5126,"VEC2")
ACC_J   = add(J4,34962,5121,"VEC4")
ACC_W   = add(W4,34962,5126,"VEC4")
ACC_IDX = add(I,34963,5125,"SCALAR")
# joint nodes
world = {n:joints[n] for n in names}
node_idx = {}
for n in names:
    loc = world[n] - (world[parent[n]] if parent[n] else 0)
    out.nodes.append(Node(name=n, translation=loc.tolist(), children=[])); node_idx[n]=len(out.nodes)-1
for n in names:
    if parent[n]: out.nodes[node_idx[parent[n]]].children.append(node_idx[n])
ibm = np.zeros((len(names),4,4),np.float32)
for k,n in enumerate(names):
    m = np.eye(4,dtype=np.float32); m[:3,3] = -world[n]; ibm[k]=m.T   # column-major
ACC_IBM = add(ibm.reshape(len(names),16),None,5126,"MAT4")
out.materials.append(Material(name="jacket", pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[0.8,0.8,0.8,1],metallicFactor=0,roughnessFactor=0.9), doubleSided=True))
out.meshes.append(Mesh(name="jacket", primitives=[Primitive(attributes=Attributes(POSITION=ACC_POS,NORMAL=ACC_NRM,TEXCOORD_0=ACC_UV,JOINTS_0=ACC_J,WEIGHTS_0=ACC_W), indices=ACC_IDX, material=0)]))
out.skins.append(Skin(name="jacket_skin", joints=[node_idx[n] for n in names], skeleton=node_idx["hips"], inverseBindMatrices=ACC_IBM))
out.nodes.append(Node(name="jacket_mesh", mesh=0, skin=0)); mesh_node=len(out.nodes)-1
out.scenes.append(Scene(nodes=[node_idx["hips"], mesh_node])); out.scene=0
out.buffers.append(Buffer(byteLength=sum(len(b) for b in bufs)))
out.set_binary_blob(b"".join(bufs)); out.save(OUT)
json.dump({n:{"pos":world[n].round(4).tolist(),"parent":parent[n]} for n in names}, open(OUT.replace(".glb","_joints.json"),"w"), indent=1)
print("wrote", OUT)
