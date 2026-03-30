from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class CostMatrix:
    C_TP: float
    C_TN: float
    C_FP: float
    C_FN: float

@dataclass
class Settings:
    seed: int = 42
    justice: str = "egalitarian"          # "egalitarian" or "rawlsian"
    stochastic: bool = True
    thresholds: int = 100                 # number of tau grid points
    sigmas: List[float] = field(default_factory=list)

    owner_costs: Optional[CostMatrix] = None
    customer_costs: Optional[CostMatrix] = None
    customer_costs_1: Optional[CostMatrix] = None

# Internal storage
_GLOBAL_SETTINGS: Optional[Settings] = None

def set_settings(settings: Settings) -> None:
    """Initialize the global settings. Call once in main()"""
    global _GLOBAL_SETTINGS
    if _GLOBAL_SETTINGS is not None:
        raise RuntimeError("GLOBAL_SETTINGS already initialized")
    _GLOBAL_SETTINGS = settings

def get_settings() -> Settings:
    """Retrieve the global settings anywhere in this process."""
    if _GLOBAL_SETTINGS is None:
        raise RuntimeError("GLOBAL_SETTINGS not initialized")
    return _GLOBAL_SETTINGS

def has_settings() -> bool:
    """Return True if GLOBAL_SETTINGS has been initialized."""
    return _GLOBAL_SETTINGS is not None

def load_config(path: str) -> Settings:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- basic fields ---
    s = Settings(
        seed=cfg.get("seed", 42),
        justice=cfg.get("justice", "egalitarian"),
        stochastic=cfg.get("stochastic", True),
        thresholds=cfg.get("thresholds", 100),
    )

    # --- sigma sweep (beta_range) ---
    if "sigmas" in cfg:
        s.sigmas = list(cfg["sigmas"])
    else:
        # explicit default, not hidden
        s.sigmas = [1_000_000, 500, 100, 50, 30, 10, 5, 2, 1, 0.5]

    # ---------------- OWNER COSTS ----------------
    if "owner_costs" not in cfg:
        raise ValueError("owner_costs must be specified in config")

    owner_cfg = cfg["owner_costs"]

    if isinstance(owner_cfg, dict) and "kind" in owner_cfg:
        # Dataset-driven owner utility (German Credit, HomeCredit, etc.)
        s.owner_costs = owner_cfg
    else:
        # Constant owner utility
        s.owner_costs = CostMatrix(**owner_cfg)


    # ---------------- CUSTOMER COSTS ----------------
    if "customer_costs" in cfg:
        s.customer_costs = CostMatrix(**cfg["customer_costs"])
        if "customer_costs_1" in cfg:
            s.customer_costs_1 = CostMatrix(**cfg["customer_costs_1"])
    else:
        # Explicitly tie customer to owner ONLY if owner is constant
        if isinstance(s.owner_costs, CostMatrix):
            s.customer_costs = CostMatrix(**owner_cfg)
        else:
            raise ValueError(
                "customer_costs must be specified when owner_costs is dataset-driven"
            )

    return s

