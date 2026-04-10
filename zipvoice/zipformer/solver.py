from typing import Optional, Union

import torch


class EulerSolver:
    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "flow_estimate",
    ):
        self.model = DiffusionModel(model=model, func_name=func_name)
        self.func_name = func_name

    def sample(
        self,
        x: torch.Tensor,
        speech_condition: torch.Tensor,
        text_condition: torch.Tensor,
        pad_mask:torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
        num_steps: int = 10,
        device: torch.device = None,
    ):

        timestep = self.time_step(t_start, t_end, num_steps, device=device)

        for step in range(num_steps):
            time_cur = timestep[step]
            time_next = timestep[step + 1]

            v_t = self.model(
                x=x,
                t=time_cur,
                speech_condition=speech_condition,
                text_condition=text_condition,
                padding_mask=pad_mask,
                device=device,
            )

            x_1 = x + (1.0 - time_cur) * v_t
            x_0 = x - time_cur * v_t

            if step < num_steps - 1:
                x = (1.0 - time_next) * x_0 + time_next * x_1
            else:
                x = x_1

        return x

    def time_step(
        self,
        t_start: float,
        t_end: float,
        num_steps: int,
        time_shift: float = 1.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        time_steps = torch.linspace(t_start, t_end, num_steps + 1).to(device)

        timesteps = time_shift * time_steps / (1 + (time_shift - 1) * time_steps)

        return timesteps


class DiffusionModel(torch.nn.Module):

    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "flow_estimate",
    ):
        super().__init__()
        self.model = model
        self.func_name = func_name
        self.model_func = getattr(self.model, func_name)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        device: torch.device = None,
    ) -> torch.Tensor:

        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(
                guidance_scale, dtype=t.dtype, device=t.device
            )

        if (guidance_scale == 0.0).all():
            return self.model_func(
                t=t,
                x_t=x,
                text_cond=text_condition,
                speech_cond=speech_condition,
                pad_mask=padding_mask,
                device=device,
            )
        else:
            assert t.dim() == 0

            x = torch.cat([x] * 2, dim=0)
            padding_mask = torch.cat([padding_mask] * 2, dim=0)

            text_condition = torch.cat(
                [torch.zeros_like(text_condition), text_condition], dim=0
            )

            if t > 0.5:
                speech_condition = torch.cat(
                    [torch.zeros_like(speech_condition), speech_condition], dim=0
                )
            else:
                guidance_scale = guidance_scale * 2
                speech_condition = torch.cat(
                    [speech_condition, speech_condition], dim=0
                )

            data_uncond, data_cond = self.model_func(
                t=t,
                x_t=x,
                text_cond=text_condition,
                speech_cond=speech_condition,
                pad_mask=padding_mask,
                device=device,
            ).chunk(2, dim=0)

            res = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
            return res
