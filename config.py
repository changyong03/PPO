from dataclasses import dataclass

@dataclass
class Config:
    env_name : str = "Pendulum-v1"
    seed: int = 42
    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    eps_clip:float = 0.2
    

    num_epochs: int = 10
    batch_size: int = 64
    horizon: int = 2048

    hidden_dim: int = 256

    