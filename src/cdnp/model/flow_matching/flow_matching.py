from typing import Optional

import torch
from loguru import logger
from torch import nn

from cdnp.model.ctx import ModelCtx
from cdnp.model.flow_matching.path.affine import CondOTProbPath
from cdnp.model.flow_matching.solver.ode_solver import ODESolver
from cdnp.model.flow_matching.utils import ModelWrapper
from cdnp.model.meta.unet import UNetModel
from cdnp.sampler.dpm_solver import (
    DPM_Solver_v3,
    NoiseScheduleFlowMatch,
    get_time_steps,
    model_wrapper,
)

SOLVER_TO_ORDER = {
    "dopri8": 8,
    "dopri5": 5,
    "bosh3": 3,
    "fehlberg2": 2,
    "adaptive_heun": 2,
    "euler": 1,
    "midpoint": 2,
    "heun2": 2,
    "heun3": 3,
    "rk4": 4,
    "explicit_adams": 4,
    "implicit_adams": 4,
    "scipy_solver": 4,
}


# TODO, get rid of the whole CFG, not needed.
class CFGScaledModel(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
        label: torch.Tensor,
        ctx: ModelCtx,
    ):
        t = torch.zeros(x.shape[0], device=x.device) + t
        if ctx.image_ctx is not None:
            x = torch.cat([x, ctx.image_ctx], dim=1)

        extra = {}
        if ctx.warmth is not None:
            extra["warmth"] = ctx.warmth
        result = self.model(x, t, extra=extra)

        self.nfe_counter += 1
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


class FlowMatching(nn.Module):
    def __init__(
        self,
        backbone: UNetModel,
        skewed_timesteps: bool,
        ode_method: str,
        skip_type: str,
        ode_opts: dict,
        num_channels: int,
        height: int,
        width: int,
        device: str,
        epsilon: float = 1e-3,
    ):
        super().__init__()
        self.skewed_timesteps = skewed_timesteps
        self.device = device
        self.path = CondOTProbPath()
        self.cfg_model = CFGScaledModel(backbone)
        self.solver = ODESolver(velocity_model=self.cfg_model)
        self.noise_schedule = NoiseScheduleFlowMatch()

        self.ode_method = ode_method
        self.skip_type = skip_type
        self.ode_opts = ode_opts
        self.epsilon = epsilon

        self.backbone = backbone
        self.num_channels = num_channels
        self.height = height
        self.width = width

    def forward(
        self,
        ctx: ModelCtx,
        trg: torch.Tensor,
        loss_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(trg)
        batch_size = trg.shape[0]

        if self.skewed_timesteps:
            t = skewed_timestep_sample(batch_size, device=self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        path_sample = self.path.sample(t=t, x_0=noise, x_1=trg)
        x_t = path_sample.x_t

        if ctx.image_ctx is not None:
            x_t = torch.cat([x_t, ctx.image_ctx], dim=1)
        t = torch.zeros(batch_size, device=self.device) + t

        extra = {}
        if ctx.warmth is not None:
            extra["warmth"] = ctx.warmth

        u_t = path_sample.dx_t

        if ctx.label_ctx:
            # TODO?
            logger.warning(
                "Conditional flow-matching generation is not yet implemented."
            )

        pred_u = self.backbone(x_t, t, extra=extra)
        loss = torch.pow(pred_u - u_t, 2)
        if loss_weight is not None:
            loss = loss * loss_weight
        return loss.mean()

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int, **kwargs) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process.
        :num_samples: Number of samples to generate.
        """

        if ctx.image_ctx is not None:
            num_samples = ctx.image_ctx.shape[0]

        shape = (num_samples, self.num_channels, self.height, self.width)

        x_T = torch.randn(*shape).to(self.device)

        if ctx.label_ctx:
            logger.warning(
                "Conditional flow-matching generation is not yet implemented."
            )

        nfe = kwargs.get("nfe", self.ode_opts["nfe"])
        ode_method = kwargs.get("ode_method", self.ode_method)
        skip_type = kwargs.get("skip_type", self.skip_type)

        if ode_method.startswith("dpm_solver_"):
            order = int(ode_method.split("_")[-1])
            return self._sample_dpm_v3(
                ctx=ctx,
                x_T=x_T,
                nfe=nfe,
                order=order,
                skip_type=skip_type,
            )
        else:
            return self._sample_odeint(
                ctx=ctx,
                x_T=x_T,
                ode_method=ode_method,
                skip_type=skip_type,
                nfe=nfe,
            )

    @torch.no_grad()
    def _sample_dpm_v3(
        self,
        ctx: ModelCtx,
        x_T: torch.Tensor,
        nfe: int,
        order: int,
        skip_type: str,
        epsilon: float = 1e-3,
    ) -> torch.Tensor:
        """
        Generates samples using the provided DPM-Solver v3 implementation.
        This version is corrected for clarity and accuracy.
        """
        solver = DPM_Solver_v3(
            noise_schedule=self.noise_schedule,
            steps=nfe,
            t_start=1.0 - self.epsilon,
            t_end=self.epsilon,
            skip_type=skip_type,
            device=self.device,
        )

        def _u_conversion(x: torch.Tensor, t: torch.Tensor, cond=None, **kwargs):
            x = x.to(torch.float32)

            # Sampler uses convention where t=1 is noise, t=0 is data.
            # FM code uses t=0 for noise, t=1 for data. Therefore, we need to flip the
            # time variable, and negate the predicted velocity field.
            t_fm = 1.0 - t
            return -self.cfg_model(x, t_fm, cfg_scale=0.0, label=None, ctx=ctx)

        wrapped_model_fn = model_wrapper(
            model=_u_conversion,
            noise_schedule=self.noise_schedule,
            model_type="flow_matching",
            guidance_type="uncond",
        )

        return solver.sample(
            x=x_T,
            model_fn=wrapped_model_fn,
            order=order,
            p_pseudo=False,
            use_corrector=False,
            c_pseudo=False,
            lower_order_final=True,
        )

    @torch.no_grad()
    def _sample_odeint(
        self,
        ctx: ModelCtx,
        x_T: torch.Tensor,
        ode_method: str,
        skip_type: str,
        nfe: int,
    ):
        order = SOLVER_TO_ORDER[ode_method]
        if nfe % order != 0:
            logger.warning(
                f"Number of steps {nfe} is not divisible by order {order}. "
                f"NFE is lower than requested. Requested: {nfe}, "
                f"actual: {nfe // order * order}."
            )
        steps = nfe // order
        time_grid = get_time_steps(
            self.noise_schedule,
            skip_type=skip_type,
            t_T=1.0 - self.epsilon,
            t_0=self.epsilon,
            N=steps,
            device=self.device,
        )
        time_grid = 1 - time_grid  # Convert to FM convention

        return self.solver.sample(
            time_grid=time_grid,
            x_init=x_T,
            method=ode_method,
            return_intermediates=False,
            atol=self.ode_opts["atol"],
            rtol=self.ode_opts["rtol"],
            step_size=self.ode_opts["step_size"],
            # TODO remove cfg, not needed
            cfg_scale=0.0,
            label=None,
            ctx=ctx,
        )

    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        return [self.sample(ctx, num_samples) for _ in range(4)]


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time
