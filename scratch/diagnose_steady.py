import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from v1_simulation.cli import load_cli_config
from v1_simulation.simulation.pipeline import V1SimulationPipeline

def diagnose():
    jax.config.update("jax_enable_x64", False)
    
    overrides = [
        "solver=diffrax_tsit5",
        "background=none",
        "model.connectivity.j=1.2",
        "training.bcm.epochs=1",
        "training.bcm.batch_size=2", # small batch for test
    ]
    
    cfg = load_cli_config(config_path=None, config_name="config", overrides=overrides)
    cfg.mode = "train"
    cfg.training.enabled = True
    
    print("Building pipeline...")
    pipeline = V1SimulationPipeline(cfg)
    pipeline.setup()
    
    # We will manually get the vector field and evaluate it at the final state of a run.
    # To do this easily, let's just run the solver and get the trajectory.
    
    print("Running solver...")
    # Get a batch from BCM data generator
    batch = next(pipeline.bcm_trainer.data_generator)
    
    # The solver returns BatchODEResult
    # But wait, diffrax_tsit5 returns only tail points by default. 
    # Let's override to get the full trajectory.
    pipeline.solver.options.store_trajectory = True
    
    result = pipeline.solver(batch)
    
    y_final_exc = result.exc_trajectory[-1] # shape (n_exc, batch)
    y_final_inh = result.inh_trajectory[-1] # shape (n_inh, batch)
    y_final = np.concatenate([y_final_exc, y_final_inh], axis=0) # shape (N, batch)
    
    print(f"Final state shape: {y_final.shape}")
    
    # Let's compute dy/dt at the final state to see how small it actually is!
    # We can just call pipeline.solver.rhs(t=3.0, y=y_final)
    # The RHS is wrapped in the jax backend, but we can access it through the model.
    # Actually, the model has `__call__(t, rates) -> drates/dt`
    
    # The V1Network __call__ expects state of shape (N, B)
    t = 3.0
    # Wait, the model might need external drive!
    # Let's just use the JAX solver's internal vector field if possible, or model.__call__
    # For a static input, external_drive(3.0) is the same as external_drive(0.0)
    
    dy = pipeline.model(t, y_final, external_drive=batch)
    dy = np.asarray(dy)
    
    f_norm_max = np.max(np.abs(dy))
    f_norm_rms = np.sqrt(np.mean(np.square(dy)))
    
    print("=======================================")
    print(f"At end of simulation (t=3.0s):")
    print(f"Max derivative norm (L_inf): {f_norm_max}")
    print(f"RMS derivative norm (L_2): {f_norm_rms}")
    print(f"Max firing rate: {np.max(y_final)}")
    print("=======================================")
    
    if f_norm_max > 1e-4:
        print("f_norm_max > 1e-4! The absolute tolerance 1e-4 is TOO STRICT for max norm.")
    else:
        print("f_norm_max < 1e-4! There must be another bug.")

if __name__ == "__main__":
    diagnose()
