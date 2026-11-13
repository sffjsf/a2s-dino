import torch
from torch.nn import functional as F

def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()

def Loss(preds, target, config):
    target = target.gt(0.5).float()
    final_prediction = resize_prediction(
        preds['final'], target
    )
    loss = IOU(torch.sigmoid(final_prediction), target)

    if config.get('only_rgb', False):
        return loss

    rgb_prediction = resize_prediction(preds['rgb'], target)
    loss = loss + config.get(
        'rgb_aux_weight', 0.2
    ) * IOU(torch.sigmoid(rgb_prediction), target)

    modal_valid = preds['modal_valid'].bool().view(-1)
    if modal_valid.any():
        modal_prediction = resize_prediction(
            preds['modal'], target
        )
        loss = loss + config.get(
            'modal_aux_weight', 0.2
        ) * IOU(
            torch.sigmoid(modal_prediction[modal_valid]),
            target[modal_valid],
        )
    return loss


def resize_prediction(prediction, target):
    return F.interpolate(
        prediction,
        size=target.shape[-2:],
        mode='bilinear',
        align_corners=False,
    )
