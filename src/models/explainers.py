import torch
import torch.nn as nn

class ReverseDistributionMnist(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.device = device

        # Model which learns the probability distribution over the pixels. ie. This generator takes a one-hot vector
        # y which encode the class we want the generator to generate, it returns the probability map over the pixels
        self.f = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 784),
                               nn.ReLU()).to(device)

    def reparametrize(self, proba, k, temperature=0.1):
        '''
        Generate samples from probability map using the gumbel reparametrization trick.
        :param proba: The probability map
        :param k: The number of pixels to sample
        :param temperature: How soft/hard our soft one hot vector are
        :return: Samples generated using k pixels and the given probability map
        '''
        # - log( - log(U) ) ~ Gumbel

        img = 0
        for _ in range(k):
            # Gumbel noise: the size
            gumbel_noise = - torch.log(-torch.log(torch.rand_like(proba))).to(self.device)

            x = (torch.log(proba) + gumbel_noise) / temperature

            sample = nn.functional.softmax(x, dim=-1)
            # img.append(sample)
            img += sample

        # img = torch.stack(img, dim=-1)
        # img, indices = torch.max(img, dim=-1)

        return img

    def forward(self, yonehot, k):
        proba = self.f(yonehot)
        return self.reparametrize(proba, k=k), proba

