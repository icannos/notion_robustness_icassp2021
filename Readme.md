## Code accompagning the paper: A generative perspective to study adversarial robustness

This is the code used to pruduce experiments that shows that it is possible to learn the generative distribution 
associated to a classifier and to evaluate distance between these distributions to assess the robustness of a model.

```
├── Readme.md                          # This file
├── requirements.txt                   # Requierements
└── src
    ├── models                        # Contains pytorch modedules to describe classifiers and generative distributions.
    │   ├── classifiers.py            # Contains classifiers models.
    │   ├── explainers.py             # Generative distribution models.
    │   ├── __init__.py
    └── run_experiments.py            # Main file to run the experiments.
```

## Usage

```
python src/run_experiments.py --help
usage: run_experiments.py [-h] [-d DIR] [-re REPEAT_EXPERIMENT] [-b BATCH_SIZE] [-cm {conv,dense}] [-ce CLASSIFIER_EPOCHS] [-aa {L2PGDAttack,LinfPGDAttack,L1PGDAttack}] [-ap ADVERSARIAL_ATTACK_PARAMETERS] [-gl {sharp,fuzzy}] [-ge GENERATIVE_EPOCHS] path

Produces experiments following the parameters in the configuration file

positional arguments:
  path                  Filepath where to store the results of the experiment.

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --dir DIR     Where to store the file of statistics produced.
  -re REPEAT_EXPERIMENT, --repeat-experiment REPEAT_EXPERIMENT
                        Number of repetition for each attack/parameter configuration (For statistical significance).
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Size of batch to use during the trainings.
  -cm {conv,dense}, --classifier-model {conv,dense}
                        Wether to use the convolutional model or the dense one.
  -ce CLASSIFIER_EPOCHS, --classifier-epochs CLASSIFIER_EPOCHS
                        Number of epoch to do while training the classifiers.
  -aa {L2PGDAttack,LinfPGDAttack,L1PGDAttack}, --adversarial-attack {L2PGDAttack,LinfPGDAttack,L1PGDAttack}
                        Name of the adversarial attack to use.
  -ap ADVERSARIAL_ATTACK_PARAMETERS, --adversarial-attack_parameters ADVERSARIAL_ATTACK_PARAMETERS
                        List of parameters to test for this adversarial attack.
  -gl {sharp,fuzzy}, --generative_loss {sharp,fuzzy}
                        Sharp or fuzzy representation.
  -ge GENERATIVE_EPOCHS, --generative-epochs GENERATIVE_EPOCHS
                        Number of epoch to do while training the generative distributions.
```