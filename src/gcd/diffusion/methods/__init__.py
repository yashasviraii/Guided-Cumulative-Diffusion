from gcd.diffusion.methods.attend_and_excite import AttendAndExciteDiffusion
from gcd.diffusion.methods.attention_modulation import AttentionModulationDiffusion
from gcd.diffusion.methods.context_switching import ContextSwitchingDiffusion
from gcd.diffusion.methods.simple_diffusion import SimpleDiffusion

METHOD_REGISTRY = {
    "simple": SimpleDiffusion,
    "attend_and_excite": AttendAndExciteDiffusion,
    "context_switching": ContextSwitchingDiffusion,
    "attention_modulation": AttentionModulationDiffusion,
}

__all__ = [
    "SimpleDiffusion",
    "AttendAndExciteDiffusion",
    "ContextSwitchingDiffusion",
    "AttentionModulationDiffusion",
    "METHOD_REGISTRY",
]
