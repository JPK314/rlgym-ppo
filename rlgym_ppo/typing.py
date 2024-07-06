from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LearnerConfig:
    n_proc: int = 8
    min_inference_size: int = 80
    render: bool = False
    render_delay: float = 0
    timestep_limit: int = 5_000_000_000
    exp_buffer_size: int = 100000
    ts_per_iteration: int = 50000
    ppo_epochs: int = 10
    ppo_batch_size: int = 50000
    ppo_minibatch_size: Optional[int] = None
    ppo_ent_coef: float = 0.005
    ppo_clip_range: float = 0.2
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    log_to_wandb: bool = False
    load_wandb: bool = True
    wandb_project_name: Optional[str] = None
    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    checkpoints_save_folder: Optional[str] = None
    add_unix_timestamp: bool = True
    checkpoint_load_folder: Optional[str] = None
    save_every_ts: int = 1_000_000
    instance_launch_delay: Optional[float] = None
    random_seed: int = 123
    n_checkpoints_to_keep: int = 5
    shm_buffer_size: int = 8192
    device: str = "auto"
    recalculate_agent_id_every_step: bool = False
    trajectory_processor_args: Dict[str, Any] = {}


@dataclass
class LearnerMetrics:
    cumulative_timesteps: int
    iteration_time: float
    timesteps_collected: int
    timestep_collection_time: float
