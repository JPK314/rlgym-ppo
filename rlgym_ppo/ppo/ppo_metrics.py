from dataclasses import dataclass


@dataclass
class PPOMetrics:
    batch_consumption_time: float
    cumulative_model_updates: int
    policy_entropy: float
    kl_divergence: float
    value_loss: float
    sb3_clip_fraction: float
    policy_update_magnitude: float
    critic_update_magnitude: float
