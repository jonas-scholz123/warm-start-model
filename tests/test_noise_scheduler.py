import pytest
import torch

from cdnp.model.noise_scheduler import CDNPScheduler


@pytest.fixture(scope="module")
def scheduler_instance():
    return CDNPScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )


@pytest.fixture
def test_data_config():
    return {"batch_size": 4, "num_features": 2, "dtype": torch.float32, "device": "cpu"}


@pytest.fixture
def base_tensors(test_data_config):
    config = test_data_config
    x_0 = torch.randn(
        config["batch_size"],
        config["num_features"],
        dtype=config["dtype"],
        device=config["device"],
    )
    mu_C = (
        torch.rand(
            config["batch_size"],
            config["num_features"],
            dtype=config["dtype"],
            device=config["device"],
        )
        * 2
        - 1
    )
    S_C_std = (
        torch.rand(
            config["batch_size"],
            config["num_features"],
            dtype=config["dtype"],
            device=config["device"],
        )
        * 0.5
        + 0.1
    )
    return x_0, mu_C, S_C_std


def test_terminal_distribution(scheduler_instance, base_tensors, test_data_config):
    _, mu_C_template, S_C_std_template = base_tensors
    config = test_data_config
    num_stat_samples = 5000  # Increased for better statistical accuracy

    # Use a single context for the statistical test (first from template)
    # And expand it to match num_stat_samples
    fixed_mu_C = mu_C_template[0:1].expand(num_stat_samples, -1)
    fixed_S_C_std = S_C_std_template[0:1].expand(num_stat_samples, -1)

    x_0_batch_stat = torch.randn(
        num_stat_samples,
        config["num_features"],
        dtype=config["dtype"],
        device=config["device"],
    )
    noise_stat = torch.randn(
        num_stat_samples,
        config["num_features"],
        dtype=config["dtype"],
        device=config["device"],
    )

    terminal_timestep_val = scheduler_instance.num_train_timesteps - 1
    timesteps_stat = torch.full(
        (num_stat_samples,),
        terminal_timestep_val,
        dtype=torch.long,
        device=config["device"],
    )

    x_T_samples = scheduler_instance.add_noise(
        x_0_batch_stat,
        noise_stat,
        timesteps_stat,
        x_T_mean=fixed_mu_C,
        x_T_std=fixed_S_C_std,
    )

    empirical_mean = x_T_samples.mean(dim=0)
    empirical_std = x_T_samples.std(dim=0)

    target_mu = fixed_mu_C[0]
    target_std = fixed_S_C_std[0]

    assert torch.allclose(empirical_mean, target_mu, atol=0.05), (
        "Mean of terminal distribution mismatch"
    )
    assert torch.allclose(empirical_std, target_std, atol=0.05), (
        "Std of terminal distribution mismatch"
    )


def test_step_oracle(scheduler_instance, base_tensors, test_data_config):
    x_0_true, mu_C, S_C_std = base_tensors
    config = test_data_config

    t_val = scheduler_instance.num_train_timesteps // 2
    timesteps_for_add_noise = torch.full(
        (config["batch_size"],), t_val, dtype=torch.long, device=config["device"]
    )

    epsilon_true = torch.randn_like(x_0_true)

    x_t_true = scheduler_instance.add_noise(
        x_0_true, epsilon_true, timesteps_for_add_noise, x_T_mean=mu_C, x_T_std=S_C_std
    )

    output = scheduler_instance.step(
        pred_eps=epsilon_true,
        timestep=t_val,
        x_t=x_t_true,
        x_T_mean=mu_C,
        x_T_std=S_C_std,
    )

    predicted_x_0_from_step = output.pred_original_sample
    assert torch.allclose(predicted_x_0_from_step, x_0_true, atol=1e-4), (
        "Oracle test: predicted x_0 mismatch"
    )
