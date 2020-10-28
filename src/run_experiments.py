from configparser import ConfigParser

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pathlib import Path

from models.classifiers import mnistConv, mnistDense
from models.explainers import ReverseDistributionMnist

from torch.optim import Adam
from advertorch.context import ctx_noparamgrad_and_eval

from advertorch.attacks import L2PGDAttack, LinfPGDAttack, L1PGDAttack

import argparse

# Fonction pour garantir la stabilité numérique (0 * log(0) = 0)
def xlogx(x):
    mask = (x > 1E-3).float()
    return (x * torch.log(x + 1E-7)) * mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produces experiments following the parameters in the configuration file")

    parser.add_argument("path", type=str, help="Filepath where to store the results of the experiment.")
    parser.add_argument("-d", "--dir", help="Where to store the file of statistics produced.", default="tmp/")

    # =================================================================================== #
    # General parameters
    # =================================================================================== #

    parser.add_argument("-re", "--repeat-experiment", type=int, default=100,
                        help="Number of repetition for each attack/parameter configuration "
                             "(For statistical significance).")

    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Size of batch to use during the trainings.")

    # =================================================================================== #
    # Classifier parameters
    # =================================================================================== #
    parser.add_argument("-cm", "--classifier-model",
                        help="Wether to use the convolutional model or the dense one.",
                        default="conv",
                        type=str,
                        choices=["conv", "dense"])

    parser.add_argument("-ce", "--classifier-epochs", type=int, default=5,
                        help="Number of epoch to do while training the classifiers.")

    # =================================================================================== #
    # Adversarial training parameters
    # =================================================================================== #

    parser.add_argument("-aa", "--adversarial-attack",
                        help="Name of the adversarial attack to use.",
                        default="LinfPGDAttack",
                        type=str,
                        choices=["L2PGDAttack", "LinfPGDAttack", "L1PGDAttack"])

    parser.add_argument("-ap", "--adversarial-attack_parameters",
                        help="List of parameters to test for this adversarial attack.",
                        type=list,
                        default=list(np.linspace(0.,1., 10)))

    # =================================================================================== #
    # Generative distributions parameters
    # =================================================================================== #

    parser.add_argument("-gl", "--generative_loss",
                        help="Sharp or fuzzy representation.",
                        default="sharp",
                        type=str,
                        choices=["sharp", "fuzzy"])

    parser.add_argument("-ge", "--generative-epochs", type=int, default=100,
                        help="Number of epoch to do while training the generative distributions.")


    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)


    device = args.device

    num_epoch = args.classifier_epochs

    adversarial_attack = eval(args.adversarial_attack)
    adv_trains = args.adversarial_attack_parameters

    stats_iter = args.repeat_experiment
    batch_size = args.batch_size

    classifier_model = args.classifier_model

    generative_loss = args.generative_loss
    generative_epochs = args.generative_epochs

    save_path = Path(args.dir) / Path(experiment)

    stats = {}

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    print(adversarial_attack.__name__)
    for adv_train in adv_trains:
        # logdir = f"trained_models/adv_train-l2{adv_train}"
        stats[adversarial_attack.__name__ + "_" + str(adv_train)] = {}
        stats[adversarial_attack.__name__ + "_" + str(adv_train)]['accs'] = []
        stats[adversarial_attack.__name__ + "_" + str(adv_train)]['acc_generator'] = []
        stats[adversarial_attack.__name__ + "_" + str(adv_train)]['proba_maps'] = []
        stats[adversarial_attack.__name__ + "_" + str(adv_train)]['models'] = []

        for iter in range(stats_iter):
            if classifier_model == "conv":
                model = mnistConv(device=device).to(device)
            elif classifier_model == "dense":
                model = mnistDense(device=device).to(device)
            else:
                exit(0)
            # We use categorical cross entropy but on log softmax
            criterion = nn.NLLLoss()
            # Adam as optimizer since it is the most used optimizer. We'll see later on if other method works better since we know
            # that adam do not generalize well on image processing problem
            optimizer = Adam(model.parameters())

            adversary = adversarial_attack(
                model, loss_fn=nn.NLLLoss(reduction="mean"), eps=adv_train,
                nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=False)

            for e in range(num_epoch):
                print(e)
                for x, y in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    bsize = y.shape[0]

                    with ctx_noparamgrad_and_eval(model):
                        adv_targeted = adversary.perturb(x, y)

                    optimizer.zero_grad()

                    loss = criterion(model(x), y) + criterion(model(adv_targeted), y)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                accs = []
                for x, y in test_loader:
                    preds = torch.argmax(model(x.to(device)), dim=1)
                    acc = (preds == y.to(device)).float().mean()

                    accs.append(acc)

                accs = torch.stack(accs)
                accs = accs.mean()

            stats[adversarial_attack.__name__ + "_" + str(adv_train)]['accs'].append(accs.cpu().detach().numpy())
            stats[adversarial_attack.__name__ + "_" + str(adv_train)]['models'].append(model.state_dict())

            # Modèle pour apprendre la distribution
            distribution = ReverseDistributionMnist(device=device)

            # optimizer
            optimizer = Adam(params=list(distribution.parameters()))

            # Entraînement du modèle génératif

            for e in range(generative_epochs):
                print(e)
                y = torch.randint(0, 10, (batch_size, 1))
                y = y.to(device)

                y_onehot = torch.zeros(batch_size, 10).to(device)
                y_onehot.scatter_(1, y, 1)

                xchap, proba = distribution(y_onehot, 500)
                xchap = xchap.view(batch_size, 1, 28, 28)
                output = model(xchap)

                p = torch.exp(output)

                img_entropy = - torch.sum(xlogx(proba).view(batch_size, -1), dim=1).mean()

                if generative_loss == "sharp":
                    loss = F.nll_loss(output, y.squeeze(), reduction='none').mean() - 0.01 * img_entropy
                elif generative_loss == "fuzzy":
                    # Only account samples that are wrongly classified
                    loss = ((1 - (torch.argmax(output, dim=1) == y.squeeze()).float()) *
                            F.nll_loss(output, y.squeeze(), reduction='none')).mean() - 0.01 * img_entropy
                else:
                    raise Exception("Bad argument for generative_loss")
                # loss = F.l1_loss(output, y_onehot, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f'{loss}')

            acc = (torch.argmax(p, dim=1) == y.squeeze().to(device)).float().mean()

            y = torch.zeros((10, 10))

            for i in range(10):
                y[i][i] = 1

            xchap, proba = distribution(y.to(device), 10)

            stats[adversarial_attack.__name__ + "_" + str(adv_train)]['acc_generator'].append(acc.detach().cpu().numpy())
            stats[adversarial_attack.__name__ + "_" + str(adv_train)]['proba_maps'].append(proba.detach().cpu().numpy())

        import pickle as pk

    pk.dump(stats, open(save_path, 'wb'), protocol=pk.HIGHEST_PROTOCOL)
