"""
GLB Retargeting Module for HY-Motion

Mirrors the FBX retarget pipeline but reads/writes glTF Binary (.glb) using
pygltflib. The math (Skeleton, retarget_animation, NPZ loader, bone mapping)
is reused verbatim from retarget_fbx.py — only the I/O layer is new.

Pipeline:
    1. Load Mixamo-rigged GLB     -> Skeleton (rest pose from joint nodes)
    2. Load NPZ motion            -> Skeleton (already in retarget_fbx.load_npz)
    3. retarget_animation()       -> ret_rots / ret_locs (reused)
    4. Write back into the GLB    -> animation.samplers/channels point at joint nodes
    5. save_binary()              -> new .glb with the original mesh / skin /
                                     materials preserved, animations replaced.

glTF conventions handled here:
    - Quaternions: glTF uses [x,y,z,w]; internal code uses [w,x,y,z].
    - TRS matrices are column-major (M @ v); we mirror retarget_fbx.load_npz's
      storage of rotation in world_matrix[:3,:3] (column-major) and translation
      in world_matrix[3,:3] (row), and use matrix_to_quaternion(R.T) for rest.
    - No PreRotation/PostRotation. q_local from retarget_animation is written
      to node.rotation samplers directly (after [w,x,y,z] -> [x,y,z,w]).
"""

from __future__ import annotations

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from pygltflib import (
        GLTF2,
        Animation,
        AnimationSampler,
        AnimationChannel,
        AnimationChannelTarget,
        Accessor,
        BufferView,
    )
    HAS_PYGLTFLIB = True
except ImportError:
    HAS_PYGLTFLIB = False

from .retarget_fbx import (
    BoneData,
    Skeleton,
    matrix_to_quaternion,
)

_TAG = "[GLB Retarget]"

# glTF accessor componentType constants
_GLTF_FLOAT = 5126


def _log(msg: str):
    print(f"{_TAG} {msg}", flush=True)


# =============================================================================
# Quaternion helpers (convention conversion)
# =============================================================================

def _quat_wxyz_to_xyzw(q):
    """[w,x,y,z] -> [x,y,z,w] (glTF convention)."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def _quat_xyzw_to_wxyz(q):
    """[x,y,z,w] (glTF) -> [w,x,y,z] (internal)."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


# =============================================================================
# glTF node -> 4x4 matrix
# =============================================================================

def _trs_to_matrix(t, r_xyzw, s) -> np.ndarray:
    """Build standard column-major 4x4 (M @ v form) from glTF TRS."""
    M = np.eye(4)
    rmat = R.from_quat(r_xyzw).as_matrix() if r_xyzw is not None else np.eye(3)
    if s is not None:
        rmat = rmat @ np.diag(s)
    M[:3, :3] = rmat
    if t is not None:
        M[:3, 3] = t
    return M


def _extract_pure_rotation(M3: np.ndarray) -> np.ndarray:
    """Strip scale from a 3x3 rotation*scale matrix. Returns pure rotation."""
    col_norms = np.linalg.norm(M3, axis=0)
    col_norms[col_norms < 1e-9] = 1.0
    return M3 / col_norms[np.newaxis, :]


def _build_parent_map(gltf) -> list[int]:
    n = len(gltf.nodes)
    parents = [-1] * n
    for i, node in enumerate(gltf.nodes):
        for c in (node.children or []):
            parents[c] = i
    return parents


# =============================================================================
# Load GLB -> Skeleton
# =============================================================================

class GLBContext:
    """Holds the loaded GLB and lookups needed when writing animation back."""

    def __init__(self):
        self.gltf = None
        self.path: str = ""
        self.binary_blob: bytes = b""
        self.skin_index: int = -1
        self.parent_node_idx: list[int] = []
        # Map from bone.name (original case) -> glTF node index
        self.bone_to_node: dict[str, int] = {}
        # Source FPS used to build the time accessor when writing
        self.write_fps: float = 30.0


def load_glb(filepath: str) -> tuple[GLBContext, Skeleton]:
    """Load a Mixamo-style GLB and build a Skeleton populated with rest pose.

    Returns (context, skeleton). The context retains a reference to the loaded
    GLTF2 object and the binary blob so that animations can be written back
    into the same file (preserving mesh / skin / materials).
    """
    if not HAS_PYGLTFLIB:
        raise ImportError(
            "pygltflib is required for GLB retargeting. Install with: pip install pygltflib"
        )

    _log(f"--- Loading target GLB ---")
    _log(f"  Path: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    _log(f"  Size: {os.path.getsize(filepath)} bytes")

    gltf = GLTF2().load(filepath)
    blob = bytes(gltf.binary_blob() or b"")
    _log(f"  Loaded: {len(gltf.nodes)} nodes, {len(gltf.skins)} skins, "
         f"{len(gltf.meshes)} meshes, {len(gltf.materials or [])} materials, "
         f"binary blob {len(blob)} bytes")

    if not gltf.skins:
        raise RuntimeError(
            f"GLB has no skin (no skeleton to retarget onto): {filepath}"
        )

    ctx = GLBContext()
    ctx.gltf = gltf
    ctx.path = filepath
    ctx.binary_blob = blob
    ctx.parent_node_idx = _build_parent_map(gltf)
    ctx.skin_index = 0
    skin = gltf.skins[0]
    if len(gltf.skins) > 1:
        _log(f"  Note: {len(gltf.skins)} skins found, using skin[0] for retargeting")

    # Forward kinematics over ALL nodes (joints + their non-joint ancestors).
    n = len(gltf.nodes)
    local_M: list[np.ndarray] = [np.eye(4)] * n
    for i, node in enumerate(gltf.nodes):
        t = node.translation if node.translation is not None else None
        r = node.rotation if node.rotation is not None else None  # XYZW
        s = node.scale if node.scale is not None else None
        local_M[i] = _trs_to_matrix(t, r, s)

    world_M: list[np.ndarray | None] = [None] * n

    def _world(i: int) -> np.ndarray:
        if world_M[i] is not None:
            return world_M[i]
        p = ctx.parent_node_idx[i]
        parent_M = np.eye(4) if p < 0 else _world(p)
        world_M[i] = parent_M @ local_M[i]
        return world_M[i]

    for i in range(n):
        _world(i)

    # Build Skeleton from joint nodes. Bone.parent_name is set to the IMMEDIATE
    # graph parent's name (so non-joint Armature ancestors are properly used by
    # retarget_animation via node_rest_rotations fallback).
    skel = Skeleton(os.path.basename(filepath))
    skel.fps = 30.0
    skel.frame_start = 0
    skel.frame_end = 0

    # Populate node_rest_rotations for ALL named nodes — retarget_animation
    # uses it as the parent-rotation fallback when the parent isn't a bone.
    for i, node in enumerate(gltf.nodes):
        wM = world_M[i]
        wR_pure = _extract_pure_rotation(wM[:3, :3])
        # matrix_to_quaternion expects FBX row-major form (it transposes
        # internally for SciPy). Pass R.T to match.
        q = matrix_to_quaternion(wR_pure.T)
        nm = node.name if node.name else f"node_{i}"
        skel.node_rest_rotations[nm] = q
        skel.all_nodes[nm] = nm

    joints = list(skin.joints)
    skipped = 0
    for node_idx in joints:
        node = gltf.nodes[node_idx]
        name = node.name if node.name else f"joint_{node_idx}"
        # Mirror load_npz / collect_skeleton_nodes case-insensitive dedup
        if name.lower() in skel.bones:
            skipped += 1
            continue

        bone = BoneData(name)
        p_idx = ctx.parent_node_idx[node_idx]
        if p_idx >= 0:
            p_node = gltf.nodes[p_idx]
            bone.parent_name = p_node.name if p_node.name else f"node_{p_idx}"
        else:
            bone.parent_name = None

        wM = world_M[node_idx]
        wR_pure = _extract_pure_rotation(wM[:3, :3])
        wT = wM[:3, 3]

        # Match load_npz storage convention exactly (see retarget_fbx.py:306-308).
        bone.world_matrix = np.eye(4)
        bone.world_matrix[:3, :3] = wR_pure
        bone.world_matrix[3, :3] = wT
        bone.local_matrix = local_M[node_idx]
        bone.head = wT.copy()
        bone.has_skeleton_attr = True
        bone.rest_rotation = matrix_to_quaternion(wR_pure.T)

        skel.add_bone(bone)
        ctx.bone_to_node[name] = node_idx

    _log(f"  Skeleton built: {len(skel.bones)} joints "
         f"({skipped} duplicate names skipped)")
    return ctx, skel


# =============================================================================
# Write animation back into the GLB
# =============================================================================

def _pad4(b: bytes) -> bytes:
    pad = (4 - (len(b) % 4)) % 4
    return b + b"\x00" * pad


def _append_accessor(
    gltf,
    blob: bytearray,
    data: np.ndarray,
    type_str: str,
    write_min_max: bool = False,
) -> int:
    """Append data as a new BufferView+Accessor at the tail of the blob.

    Returns the new accessor index. componentType is FLOAT (5126).
    """
    raw = data.tobytes()
    byte_offset = len(blob)
    blob.extend(_pad4(raw))

    bv = BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(raw))
    if gltf.bufferViews is None:
        gltf.bufferViews = []
    gltf.bufferViews.append(bv)
    bv_index = len(gltf.bufferViews) - 1

    acc = Accessor(
        bufferView=bv_index,
        byteOffset=0,
        componentType=_GLTF_FLOAT,
        count=int(data.shape[0]),
        type=type_str,
    )
    if write_min_max:
        if data.ndim == 1:
            acc.min = [float(data.min())]
            acc.max = [float(data.max())]
        else:
            acc.min = data.min(axis=0).astype(float).tolist()
            acc.max = data.max(axis=0).astype(float).tolist()

    if gltf.accessors is None:
        gltf.accessors = []
    gltf.accessors.append(acc)
    return len(gltf.accessors) - 1


def apply_retargeted_animation_glb(
    ctx: GLBContext,
    tgt_skel: Skeleton,
    ret_rots: dict,
    ret_locs: dict,
    frame_start: int,
    frame_end: int,
    fps: float = 30.0,
):
    """Replace the GLB's animations with one new clip driven by ret_rots/ret_locs.

    ret_rots: { bone_name: { frame: [w,x,y,z] local quat } }
    ret_locs: { bone_name: { frame: [x,y,z] local translation } } (root only)
    """
    gltf = ctx.gltf
    n_frames = frame_end - frame_start + 1
    ctx.write_fps = float(fps)

    _log(f"--- Writing animation into GLB ---")
    _log(f"  Frames: {frame_start}..{frame_end} ({n_frames}), fps={fps}")
    _log(f"  Channels requested: {len(ret_rots)} rotation, {len(ret_locs)} translation")

    # Reset animations (orphaned old bufferViews/accessors are tolerated by glTF
    # readers; cleaning them up would require a remap pass that's not worth it).
    gltf.animations = []

    blob = bytearray(ctx.binary_blob)

    times = np.array(
        [(f - frame_start) / float(fps) for f in range(frame_start, frame_end + 1)],
        dtype=np.float32,
    )
    time_acc = _append_accessor(gltf, blob, times, "SCALAR", write_min_max=True)

    samplers: list[AnimationSampler] = []
    channels: list[AnimationChannel] = []

    def _resolve_node(bone_name: str) -> int | None:
        if bone_name in ctx.bone_to_node:
            return ctx.bone_to_node[bone_name]
        # Fallback: case-insensitive lookup over bone_to_node
        lo = bone_name.lower()
        for k, v in ctx.bone_to_node.items():
            if k.lower() == lo:
                return v
        return None

    # Rotations
    rot_written = 0
    rot_skipped = 0
    for bone_name, rots in ret_rots.items():
        node_idx = _resolve_node(bone_name)
        if node_idx is None:
            rot_skipped += 1
            continue
        out = np.zeros((n_frames, 4), dtype=np.float32)
        for i, f in enumerate(range(frame_start, frame_end + 1)):
            q = rots.get(f)
            if q is None:
                q = np.array([1.0, 0.0, 0.0, 0.0])
            out[i] = _quat_wxyz_to_xyzw(q)
        out_acc = _append_accessor(gltf, blob, out, "VEC4")
        samplers.append(
            AnimationSampler(input=time_acc, output=out_acc, interpolation="LINEAR")
        )
        channels.append(
            AnimationChannel(
                sampler=len(samplers) - 1,
                target=AnimationChannelTarget(node=node_idx, path="rotation"),
            )
        )
        rot_written += 1

    # Translations (typically root only)
    loc_written = 0
    loc_skipped = 0
    for bone_name, locs in ret_locs.items():
        node_idx = _resolve_node(bone_name)
        if node_idx is None:
            loc_skipped += 1
            continue
        out = np.zeros((n_frames, 3), dtype=np.float32)
        for i, f in enumerate(range(frame_start, frame_end + 1)):
            v = locs.get(f)
            if v is None:
                v = np.zeros(3)
            out[i] = v
        out_acc = _append_accessor(gltf, blob, out, "VEC3")
        samplers.append(
            AnimationSampler(input=time_acc, output=out_acc, interpolation="LINEAR")
        )
        channels.append(
            AnimationChannel(
                sampler=len(samplers) - 1,
                target=AnimationChannelTarget(node=node_idx, path="translation"),
            )
        )
        loc_written += 1

    if rot_skipped or loc_skipped:
        _log(f"  Unmapped channels skipped: {rot_skipped} rotation, "
             f"{loc_skipped} translation")
    _log(f"  Wrote {rot_written} rotation + {loc_written} translation channels")

    if not channels:
        _log("  WARNING: no animation channels written — output GLB will be static.")

    gltf.animations = [
        Animation(name="Take 001", samplers=samplers, channels=channels)
    ]

    # Update buffer length and re-set the binary blob.
    if gltf.buffers:
        gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    ctx.binary_blob = bytes(blob)


def save_glb(ctx: GLBContext, output_path: str) -> str:
    """Save the (animated) GLB. Returns the final path."""
    if not output_path.lower().endswith(".glb"):
        output_path = output_path + ".glb"
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ctx.gltf.save_binary(output_path)
    if os.path.exists(output_path):
        _log(f"Saved: {output_path} ({os.path.getsize(output_path)} bytes)")
    else:
        _log(f"WARNING: file not found after save: {output_path}")
    return output_path
