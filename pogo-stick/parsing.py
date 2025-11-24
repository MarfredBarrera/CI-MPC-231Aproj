"""load in URDFs with BRAX and provide interface with necessary contact forces"""

import brax

# Load URDF directly into MuJoCo model
path = "urdfs/Hound_new_heavy/Hound_model.xml"
# mj_model = mujoco.MjModel.from_xml_path(path)
# mj_data = mujoco.MjData(mj_model)
# Initialize the MJX pipeline with the MuJoCo model
# MJX functions (step, init) will use this model structure
# pipeline_env = pipeline.Pipeline(mj_model, mj_data)
import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.mjx import pipeline
from mujoco import mjx

# NOTE this is how we convert a URDF to a MJCF file - then we manually add actuators after!
# # 1. Load your URDF (even with nu=0)
# model = mujoco.MjModel.from_xml_path(path)

# # 2. Save it as a standard MJCF file
# # This creates a valid MuJoCo XML file with all your meshes and links converted
# mujoco.mj_saveLastXML("urdfs/Hound_new_heavy/Hound_model.xml", model)

# print("Conversion complete! Open 'hound_converted.xml' to edit.")


class UrdfEnv(PipelineEnv):
    def __init__(self, xml_path, backend="mjx", **kwargs):
        # Load URDF to MjModel
        mj_model = mujoco.MjModel.from_xml_path(xml_path)

        # PATCH: Convert all Cylinders (5) to Capsules (3)
        # Iterate over all geometries
        for i in range(mj_model.ngeom):
            # If the geometry type is Cylinder (5)
            if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                print(f"Converting geom {i} from CYLINDER to CAPSULE to fix MJX crash.")
                mj_model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE

        self.mjx_model = mjx.put_model(mj_model)

        # Initialize the PipelineEnv (handles MJX conversion internally)
        super().__init__(self.mjx_model, backend=backend, n_frames=1, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        # You must implement reset logic
        q = self.sys.qpos0
        qd = jax.numpy.zeros(self.sys.nv)
        pipeline_state = self.pipeline_init(q, qd)
        return State(pipeline_state, jax.numpy.zeros(1), jax.numpy.zeros(1), {})

    def step(self, state: State, action: jax.Array) -> State:
        # Stepping uses the backend (mjx) automatically
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        return state.replace(pipeline_state=pipeline_state)


# Usage
env = UrdfEnv(path)
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

state = reset_fn(jax.random.PRNGKey(0))
action = jax.numpy.zeros(env.sys.nu)
next_state = step_fn(state, action)

# step_fn(reset_fn(jax.random.PRNGKey(0)))

pass
