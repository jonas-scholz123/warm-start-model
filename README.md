# cdnp

Create a .env file with the following contents:

```
WANDB_API_KEY=<Your API Key>
```

## Running the code

### Command line config groups

```
mode in {"dev", "prod"},
data in {"cifar10", "mnist"},
```

#### Mode

`dev` is a dry run for testing that everything is working as expected. Only a small amount of data is loaded, and the model training is broken after one iteration. Results are not sent to wandb.

E.g.

```
python src/cdnp/train.py mode=dev
```

for a dry run and

```
python src/cdnp/train.py mode=prod
```

for a real training run.

#### Data

Determines which data source to use. Options are `cifar10` and `mnist`.

- `cifar10`: Use the CIFAR-10 dataset.
- `mnist`: Use the MNIST dataset.
