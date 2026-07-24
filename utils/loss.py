import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.softmax(logits, dim=1)

        targets = targets.clone()
        targets[targets == 255] = 0

        targets = F.one_hot(
                 targets.long(),
                 num_classes=logits.shape[1]
                           ).permute(0, 3, 1, 2).float()

        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        ce_loss = self.ce(logits, targets)

        dice_loss = self.dice(logits, targets)

        total_loss = ce_loss + dice_loss

        return total_loss