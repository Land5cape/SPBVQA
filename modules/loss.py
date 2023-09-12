import torch


class MyHuberLoss(torch.nn.Module):
    def __init__(self):
        super(MyHuberLoss, self).__init__()
        self.delta = 0.4

    # Pseudo-Huber loss
    def pseudo_huber_loss(self, pred, target, delta):
        return delta ** 2 * ((1 + ((pred - target) / delta) ** 2) ** 0.5 - 1)


    def forward(self, output, label):
        loss = self.pseudo_huber_loss(output, label, self.delta)
        loss = torch.mean(loss)
        return loss