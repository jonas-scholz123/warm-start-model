import torch
from loguru import logger
from torch import nn

from cdnp.model.ctx import ModelCtx
from cdnp.model.flow_matching.path.affine import CondOTProbPath
from cdnp.model.flow_matching.solver.ode_solver import ODESolver
from cdnp.model.flow_matching.utils import ModelWrapper


# TODO, get rid of the whole CFG, not needed.
class CFGScaledModel(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
    ):
        t = torch.zeros(x.shape[0], device=x.device) + t
        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


class FlowMatching(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        skewed_timesteps: bool,
        edm_schedule: bool,
        ode_method: str,
        ode_opts: dict,
        num_channels: int,
        height: int,
        width: int,
        device: str,
    ):
        super().__init__()
        self.skewed_timesteps = skewed_timesteps
        self.device = device
        self.path = CondOTProbPath()
        self.cfg_model = CFGScaledModel(backbone)
        self.solver = ODESolver(velocity_model=self.cfg_model)

        self.ode_method = ode_method
        self.ode_opts = ode_opts

        self.backbone = backbone
        self.num_channels = num_channels
        self.height = height
        self.width = width

        # TODO: Dependency inject these
        if edm_schedule:
            self.time_grid = get_time_discretization(nfes=ode_opts["nfe"])
        else:
            self.time_grid = torch.tensor([0.0, 1.0], device=device)

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(trg)
        batch_size = trg.shape[0]

        if self.skewed_timesteps:
            t = skewed_timestep_sample(batch_size, device=self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        path_sample = self.path.sample(t=t, x_0=noise, x_1=trg)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        if ctx.label_ctx:
            logger.warning(
                "Conditional flow-matching generation is not yet implemented."
            )

        # TODO
        pred_u = self.backbone(x_t, t, extra={})
        return torch.pow(pred_u - u_t, 2).mean()

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process.
        :num_samples: Number of samples to generate.
        """

        shape = (num_samples, self.num_channels, self.height, self.width)

        x_T = torch.randn(*shape).to(self.device)

        if ctx.label_ctx:
            logger.warning(
                "Conditional flow-matching generation is not yet implemented."
            )

        dummy_labels = torch.zeros(num_samples, dtype=torch.long, device=self.device)

        return self.solver.sample(
            time_grid=self.time_grid,
            x_init=x_T,
            method=self.ode_method,
            return_intermediates=False,
            atol=self.ode_opts["atol"],
            rtol=self.ode_opts["rtol"],
            step_size=self.ode_opts["step_size"],
            # TODO remove cfg, not needed
            cfg_scale=0.0,
            label=dummy_labels,
        )

    def make_plot(self, ctx: ModelCtx) -> list[torch.Tensor]:
        return [self.sample(ctx, 4)]


def get_time_discretization(nfes: int, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    return t_samples


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time
