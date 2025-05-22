import pytest
import torch
from mlbnb.types import Split
from torch.utils.data import Dataset

from cdnp.data.mnist import MnistDataset
from cdnp.task.inpainting import InpaintingDataset
from cdnp.util.instantiate import load_config

ClassificationSample = tuple[torch.Tensor, torch.Tensor]


class DummyClassificationDataset(Dataset[ClassificationSample]):
    def __init__(
        self,
        num_samples: int,
        image_shape: tuple[int, ...],
        value_range: tuple[float, float] = (0, 1),
    ):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.value_range = value_range
        self.data = []
        for i in range(num_samples):
            # Create images with values from 0 to num_samples-1 for easy identification
            img = torch.full(
                image_shape,
                float(i + 1) * (value_range[1] - value_range[0]) / num_samples
                + value_range[0],
            )
            label = torch.tensor(i % 10)  # Dummy labels
            self.data.append((img, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> ClassificationSample:
        if idx >= self.num_samples:
            raise IndexError("Index out of bounds")
        return self.data[idx]


# --- Pytest Fixtures ---
@pytest.fixture
def delegate_dataset():
    cfg = load_config()
    return MnistDataset(cfg.paths, 0.1, Split.TRAIN, torch.Generator(), (0.5,), (0.5,))


@pytest.fixture
def small_delegate_dataset():
    return DummyClassificationDataset(
        num_samples=2, image_shape=(1, 2, 2), value_range=(10, 20)
    )


@pytest.fixture
def seeded_generator():
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


@pytest.fixture
def another_seeded_generator():
    gen = torch.Generator()
    gen.manual_seed(1234)
    return gen


def test_inpainting_dataset_initialization_and_len(delegate_dataset, seeded_generator):
    """Test basic initialization and length."""
    inp_dataset = InpaintingDataset(
        delegate=delegate_dataset, min_frac=0.1, max_frac=0.5, gen=seeded_generator
    )
    assert len(inp_dataset) == len(delegate_dataset), (
        "Length should match delegate dataset"
    )
    assert inp_dataset.min_frac == 0.1
    assert inp_dataset.max_frac == 0.5
    assert inp_dataset.gen == seeded_generator


def test_getitem_return_type_and_shape(small_delegate_dataset, seeded_generator):
    """Test the return type and shapes of the tensors from __getitem__."""
    inp_dataset = InpaintingDataset(
        delegate=small_delegate_dataset,
        min_frac=0.2,
        max_frac=0.8,
        gen=seeded_generator,
    )
    original_x, _ = small_delegate_dataset[0]
    x_masked, mask = inp_dataset[0]

    assert isinstance(x_masked, torch.Tensor), "x_masked should be a torch.Tensor"
    assert isinstance(mask, torch.Tensor), "mask should be a torch.Tensor"
    assert x_masked.shape == original_x.shape, (
        "x_masked shape should match original image shape"
    )
    assert mask.shape == original_x.shape, (
        "mask shape should match original image shape"
    )
    assert mask.dtype == torch.bool, "Mask should be a boolean tensor"


def test_mask_application(small_delegate_dataset, seeded_generator):
    """Test if the mask is correctly applied to x."""
    inp_dataset = InpaintingDataset(
        delegate=small_delegate_dataset,
        min_frac=0.3,  # Arbitrary values
        max_frac=0.7,
        gen=seeded_generator,
    )
    original_x, _ = small_delegate_dataset[
        0
    ]  # Original image has values 10.0 to 10.5 for sample 0
    x_masked, mask = inp_dataset[0]

    # Where mask is False (0), x_masked should be 0
    assert torch.all(x_masked[~mask] == 0.0).item(), (
        "Elements where mask is False should be 0"
    )
    # Where mask is True (1), x_masked should be original_x
    assert torch.all(x_masked[mask] == original_x[mask]).item(), (
        "Elements where mask is True should match original"
    )


def test_mask_proportion_extreme_cases(small_delegate_dataset):
    """Test extreme cases for min_frac and max_frac: all masked or none masked."""
    gen1 = torch.Generator().manual_seed(10)
    inp_dataset_all_masked_out = InpaintingDataset(
        delegate=small_delegate_dataset,
        min_frac=0.0,
        max_frac=0.0,  # frac will be 0.0
        gen=gen1,
    )
    x_masked_all_out, mask_all_out = inp_dataset_all_masked_out[0]
    assert torch.all(~mask_all_out).item(), "With frac=0, mask should be all False"
    assert torch.all(x_masked_all_out == 0.0).item(), (
        "With frac=0, x_masked should be all zeros"
    )

    gen2 = torch.Generator().manual_seed(20)  # Different generator instance
    inp_dataset_none_masked_out = InpaintingDataset(
        delegate=small_delegate_dataset,
        min_frac=1.0,
        max_frac=1.0,  # frac will be 1.0
        gen=gen2,
    )
    original_x, _ = small_delegate_dataset[0]
    x_masked_none_out, mask_none_out = inp_dataset_none_masked_out[0]
    assert torch.all(mask_none_out).item(), "With frac=1, mask should be all True"
    assert torch.all(x_masked_none_out == original_x).item(), (
        "With frac=1, x_masked should be original x"
    )


def test_generator_reproducibility(small_delegate_dataset):
    """Test that using the same seeded generator yields the same results."""
    gen1_seed42 = torch.Generator().manual_seed(42)
    inp_dataset1 = InpaintingDataset(
        delegate=small_delegate_dataset, min_frac=0.2, max_frac=0.7, gen=gen1_seed42
    )
    x_masked1_idx0, mask1_idx0 = inp_dataset1[0]
    # Reset generator state for the second dataset to be identical
    gen2_seed42 = torch.Generator().manual_seed(42)
    inp_dataset2 = InpaintingDataset(
        delegate=small_delegate_dataset, min_frac=0.2, max_frac=0.7, gen=gen2_seed42
    )
    x_masked2_idx0, mask2_idx0 = inp_dataset2[0]

    assert torch.equal(x_masked1_idx0, x_masked2_idx0), (
        "x_masked should be identical with same seeded generator for same index"
    )
    assert torch.equal(mask1_idx0, mask2_idx0), (
        "mask should be identical with same seeded generator for same index"
    )

    # Check for a different index to ensure generator advances
    gen1_seed42_reset = torch.Generator().manual_seed(42)  # For inp_dataset1
    inp_dataset1_re = InpaintingDataset(
        delegate=small_delegate_dataset,
        min_frac=0.2,
        max_frac=0.7,
        gen=gen1_seed42_reset,
    )
    _, _ = inp_dataset1_re[0]  # Call once to advance generator
    x_masked1_idx1, mask1_idx1 = inp_dataset1_re[1]

    gen2_seed42_reset = torch.Generator().manual_seed(42)  # For inp_dataset2
    inp_dataset2_re = InpaintingDataset(
        delegate=small_delegate_dataset,
        min_frac=0.2,
        max_frac=0.7,
        gen=gen2_seed42_reset,
    )
    _, _ = inp_dataset2_re[0]  # Call once to advance generator
    x_masked2_idx1, mask2_idx1 = inp_dataset2_re[1]

    assert torch.equal(x_masked1_idx1, x_masked2_idx1), (
        "x_masked should be identical for index 1 as well"
    )
    assert torch.equal(mask1_idx1, mask2_idx1), (
        "mask should be identical for index 1 as well"
    )

    # Ensure they are different from index 0
    assert not torch.equal(x_masked1_idx0, x_masked1_idx1), (
        "Output for index 0 and 1 should differ"
    )


def test_generator_non_reproducibility_different_seeds(
    small_delegate_dataset, seeded_generator, another_seeded_generator
):
    """Test that different seeded generators yield different results."""
    inp_dataset1 = InpaintingDataset(
        delegate=small_delegate_dataset,
        min_frac=0.2,
        max_frac=0.7,
        gen=seeded_generator,  # seed 42
    )
    x_masked1, mask1 = inp_dataset1[0]

    inp_dataset2 = InpaintingDataset(
        delegate=small_delegate_dataset,
        min_frac=0.2,
        max_frac=0.7,
        gen=another_seeded_generator,  # seed 1234
    )
    x_masked2, mask2 = inp_dataset2[0]

    # With very high probability, these will be different for non-trivial images/frac range
    assert not torch.equal(x_masked1, x_masked2), (
        "x_masked should differ with different seeded generators"
    )
    assert not torch.equal(mask1, mask2), (
        "mask should differ with different seeded generators"
    )
