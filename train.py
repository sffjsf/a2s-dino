import sys
import os
import time
import random
import cv2
from math import exp
from par import PAR
from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data import Train_Dataset, Test_Dataset, get_loader, get_test_list
from test import test_model
import torch
from torch.nn import utils
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from base.framework_factory import load_framework

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

STAGE1_DINO_FG_THRESHOLD = 0.8
STAGE1_DINO_BG_THRESHOLD = 0.2
STAGE1_DINO_ENTROPY_THRESHOLD = 0.65
STAGE1_DINO_TOPK_RATIO = 0.1

torch.set_printoptions(precision=5)

_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)

def stage1_dino_affinity_prior_loss(preds, tokens, config):
    eps = 1e-6
    B, _, token_h, token_w = tokens.shape
    N = token_h * token_w

    valid_sample = tokens.abs().sum(dim=(1, 2, 3)).gt(0)
    if valid_sample.sum() == 0:
        return tokens.new_zeros(())

    token_flat = tokens.flatten(2).transpose(1, 2)
    token_flat = F.normalize(token_flat, dim=-1)
    token_sim = torch.bmm(token_flat, token_flat.transpose(1, 2))
    token_sim = (token_sim + 1) / 2

    logits_token = F.interpolate(
        preds[0], size=(token_h, token_w), mode='bilinear', align_corners=True
    )
    pred_token = torch.sigmoid(logits_token)
    pred_seed = pred_token.detach().flatten(2)

    fg_seed_mask = pred_seed.ge(STAGE1_DINO_FG_THRESHOLD).float()
    bg_seed_mask = pred_seed.le(STAGE1_DINO_BG_THRESHOLD).float()
    min_seed = min(N, max(1, int(N * STAGE1_DINO_TOPK_RATIO)))
    fg_seed_count = fg_seed_mask.sum(dim=2, keepdim=True)
    if fg_seed_count.lt(1).any():
        topk_idx = torch.topk(pred_seed, k=min_seed, dim=2).indices
        topk_mask = torch.zeros_like(fg_seed_mask).scatter_(2, topk_idx, 1.0)
        fg_seed_mask = torch.where(fg_seed_count.lt(1), topk_mask, fg_seed_mask)

    bg_seed_count = bg_seed_mask.sum(dim=2, keepdim=True)
    if bg_seed_count.lt(1).any():
        bottomk_idx = torch.topk(1 - pred_seed, k=min_seed, dim=2).indices
        bottomk_mask = torch.zeros_like(bg_seed_mask).scatter_(2, bottomk_idx, 1.0)
        bg_seed_mask = torch.where(bg_seed_count.lt(1), bottomk_mask, bg_seed_mask)

    fg_seed_weight = pred_seed * fg_seed_mask
    bg_seed_weight = (1 - pred_seed) * bg_seed_mask
    fg_prior = torch.bmm(fg_seed_weight, token_sim) / (fg_seed_weight.sum(dim=2, keepdim=True) + eps)
    bg_prior = torch.bmm(bg_seed_weight, token_sim) / (bg_seed_weight.sum(dim=2, keepdim=True) + eps)
    prior = torch.clamp(fg_prior * (1 - bg_prior), 0, 1)
    prior = prior.view(B, 1, token_h, token_w)
    prior_min = prior.amin(dim=(2, 3), keepdim=True)
    prior_max = prior.amax(dim=(2, 3), keepdim=True)
    prior = (prior - prior_min) / (prior_max - prior_min + eps)

    p = torch.clamp(pred_seed, eps, 1 - eps)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)) / torch.log(p.new_tensor(2.0))
    high_entropy = entropy.ge(STAGE1_DINO_ENTROPY_THRESHOLD).float().view(B, 1, token_h, token_w)
    low_entropy = 1 - high_entropy
    stable_target = pred_seed.ge(0.5).float().view(B, 1, token_h, token_w)
    target = low_entropy * stable_target + high_entropy * prior.detach()

    loss = F.mse_loss(
        pred_token, target.detach(), reduction='none'
    )
    loss = loss.mean(dim=(1, 2, 3))
    loss = loss * valid_sample.float()
    return loss.sum() / (valid_sample.float().sum() + eps)



def token_to_rgb(token):
    """Project BxCxHxW tokens to RGB with per-sample PCA for TensorBoard."""
    visualizations = []
    for sample in token.detach().float().cpu():
        channels, height, width = sample.shape
        points = sample.reshape(channels, -1).t()
        points = points - points.mean(dim=0, keepdim=True)
        # PCA is for visualization only; fall back to the first channels if needed.
        try:
            _, _, basis = torch.pca_lowrank(points, q=3, center=False)
            rgb = points.matmul(basis[:, :3]).t().reshape(3, height, width)
        except RuntimeError:
            rgb = sample[:3]
        low = rgb.flatten(1).amin(dim=1).view(3, 1, 1)
        high = rgb.flatten(1).amax(dim=1).view(3, 1, 1)
        visualizations.append((rgb - low) / (high - low).clamp_min(1e-6))
    return torch.stack(visualizations)


def log_token_alignment(writer, images, tokens, flips, step, max_images):
    count = min(max_images, images.size(0))
    rgb = images[:count].detach().float().cpu()
    mean = rgb.new_tensor(_IMAGE_MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(_IMAGE_STD).view(1, 3, 1, 1)
    rgb = (rgb * std + mean).clamp(0, 1)

    token_rgb = token_to_rgb(tokens[:count])
    token_rgb = F.interpolate(
        token_rgb, size=rgb.shape[-2:], mode='bilinear', align_corners=False
    )
    current_pairs = torch.cat((rgb, token_rgb), dim=3)

    restored_rgb = rgb.clone()
    restored_token = token_rgb.clone()
    flip_flags = [bool(flag) for flag in flips[:count]]
    for index, was_flipped in enumerate(flip_flags):
        if was_flipped:
            restored_rgb[index] = restored_rgb[index].flip(-1)
            restored_token[index] = restored_token[index].flip(-1)
    restored_pairs = torch.cat((restored_rgb, restored_token), dim=3)

    writer.add_image(
        'token_alignment/model_input_rgb_left_token_right',
        make_grid(current_pairs, nrow=1), step
    )
    writer.add_image(
        'token_alignment/restored_rgb_left_token_right',
        make_grid(restored_pairs, nrow=1), step
    )
    writer.add_text(
        'token_alignment/flip_flags',
        ', '.join('{}: {}'.format(i, flag) for i, flag in enumerate(flip_flags)),
        step
    )

def main():
    # Loading model
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
        
    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    par = PAR(dilations=[1,2,4,8],num_iter=6)
    par.cuda()
    config['net_name'] = net_name
    stage = config['stage']
    writer = None
    if config.get('use_offline_token', False):
        log_dir = os.path.join(
            config['tensorboard_dir'], net_name, config['backbone'], config['sub']
        )
        writer = SummaryWriter(log_dir=log_dir)
        print('TensorBoard logs: {}'.format(log_dir))
    
    if config['weight'] != '':
        print('Load weights from: {}.'.format(config['weight']))
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'),strict=False)
        
    train_loader = get_loader(config)
    
    test_sets = get_test_list(config['vals'], config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    ave_batch = config['ave_batch']
    #batch = ave_batch * config['batch']
    trset = config['trset']
    batch_idx = 0
    model.zero_grad()
    for epoch in range(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        
        if debug:
            test_model(model, test_sets, config, epoch)
        
        bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=num_iter)
        
        st = time.time()
        loss_count, adb_count, ac_count, mse_count = 0, 0, 0, 0
        optim.zero_grad()
        
        fin_lr = 0.2
        for i, pack in enumerate(train_loader, start=1):
            cur_it = i + (epoch-1) * num_iter
            total_it = num_epoch * num_iter
            itr = (1 - cur_it / total_it) * (1 - fin_lr) + fin_lr
            mul = itr
            
            if stage == 1:
                tune = 2 if 'MSB-TR' in pack['name'][0] else 20
                if epoch > tune:
                    optim.param_groups[0]['lr'] = config['lr'] * mul * 0.01
                else:
                    optim.param_groups[0]['lr'] = 0
                
                optim.param_groups[1]['lr'] = config['lr'] * mul
            else:
                if epoch == 1:
                    optim.param_groups[0]['lr'] = 0
                else:
                    optim.param_groups[0]['lr'] = config['lr'] * mul * 0.1
                optim.param_groups[1]['lr'] = config['lr'] * mul
                
            if config['finetune']:
                optim.param_groups[0]['lr'] = 0
                optim.param_groups[1]['lr'] = 0
                optim.param_groups[2]['lr'] = config['lr'] * mul
                    
            images = pack['image'].float()
            gts = pack['gt'].float()
            modals = pack['modal'].float()
            modal_types = pack['modal_type'].long()
            gt_names = pack['name']
            flips = pack['flip']
            
            images, gts, modals = images.cuda(), gts.cuda(), modals.cuda()
            modal_types = modal_types.cuda()
            tokens = pack['token'].float().cuda() if config.get('use_offline_token', False) else None
            
            priors = [images]

            #priors.append(images)
            loss = 0
            if stage == 1:
                if 'dep' in pack.keys():
                    priors.append(pack['dep'].float().cuda())
                if 'of' in pack.keys():
                    priors.append(pack['of'].float().cuda())
                if 'th' in pack.keys():
                    priors.append(pack['th'].float().cuda())
                # Stage 1 uses tokens only in the DINO prior loss; A2S itself
                # still receives the RGB tensor exactly as before.
                Y = model(images, 'train')

                config['param'] = tran_param(config)
                images_temp = transform(images, False, config)
                priors = torch.cat(priors, dim=1)
                priors_temp = transform(priors, False, config)
                
                Y_ref = model(images_temp, 'train')
                
                if net_name == 'cornet':
                    gts_temp = transform(gts, True, config)
                    loss0 = model_loss(Y, gts, config)
                    loss1 = model_loss(Y_ref, gts_temp, config)
                    loss2 = F.mse_loss(
                        torch.sigmoid(Y_ref['final']),
                        transform(torch.sigmoid(Y['final']).detach(), False, config)
                    )
                else:
                    lr_weight = np.array(config['lrw'].split(',')).astype(np.float32)
                    if lr_weight is None or len(lr_weight) != 3:
                        lr_weight = [0.5, 0.05, 1]
                    loss0, loss1, loss2 = model_loss(
                        Y, priors, Y_ref, priors_temp, epoch, lr_weight, config, gt_names
                    )
                loss += loss0 + loss1 + loss2
                dino_raw_loss = torch.zeros((), device=images.device)
                loss3 = torch.zeros((), device=images.device)
                if config.get('use_stage1_dino_prior', False) and tokens is not None and epoch > 1:
                    dino_raw_loss = stage1_dino_affinity_prior_loss(Y['sal'], tokens, config)
                    loss3 = dino_raw_loss * config.get('stage1_dino_weight', 1)
                    loss += loss3

                if writer is not None and cur_it % config['tb_scalar_interval'] == 0:
                    writer.add_scalar('stage1/total', float(loss.detach()), cur_it)
                    writer.add_scalar('stage1/csd', float(loss0.detach()), cur_it)
                    writer.add_scalar('stage1/btm', float(loss1.detach()), cur_it)
                    writer.add_scalar('stage1/mse', float(loss2.detach()), cur_it)
                    writer.add_scalar('stage1/dino_raw', float(dino_raw_loss.detach()), cur_it)
                    writer.add_scalar('stage1/dino_weighted', float(loss3.detach()), cur_it)
                    for group_index, group in enumerate(optim.param_groups):
                        writer.add_scalar(
                            'learning_rate/group_{}'.format(group_index), group['lr'], cur_it
                        )

                ac_count += loss1
                mse_count += loss2
                
            elif stage > 1:
                priors.append(modals)
                if net_name == 'cornet':
                    if config['only_rgb']:
                        model_input = [images, tokens]
                    else:
                        model_input = [
                            images, modals, tokens, modal_types
                        ]
                else:
                    model_input = priors
                Y = model(model_input, 'train')

                config['param'] = tran_param(config)
                images_t = transform(images, False, config)
                tokens_t = transform(tokens, False, config) if tokens is not None else None
                modals_t = transform(modals, False, config)
                gts_t = transform(gts, True, config)
                Y_t = transform(Y['final'].detach(), False, config)
                
                priors_t = [images_t]
                priors_t.append(modals_t)
                
                if net_name == 'cornet':
                    if config['only_rgb']:
                        ref_input = [images_t, tokens_t]
                    else:
                        ref_input = [
                            images_t,
                            modals_t,
                            tokens_t,
                            modal_types,
                        ]
                else:
                    ref_input = priors_t
                Y_ref = model(ref_input, 'train')
                #print(images.shape,images_t.shape)
                #print(Y['final'].shape,Y_ref['final'].shape,Y_t.shape)
                if net_name == 'cornet':
                    loss0 = model_loss(Y, gts, config)
                    loss1 = model_loss(Y_ref, gts_t, config)
                    loss2 = F.mse_loss(torch.sigmoid(Y_ref['final']), torch.sigmoid(Y_t))
                else:
                    loss0, loss1, loss2 = model_loss(
                        Y, gts.gt(0.5).float(), Y_ref,
                        gts_t.gt(0.5).float(), Y_t, config
                    )
                loss = loss0+loss1+loss2
                ac_count += loss1
                mse_count += loss2

                # TensorBoard is only for the stage-2 token-fusion network.
                if writer is not None and cur_it % config['tb_scalar_interval'] == 0:
                    writer.add_scalar('loss/total', float(loss.detach()), cur_it)
                    writer.add_scalar('loss/loss0', float(loss0.detach()), cur_it)
                    writer.add_scalar('loss/loss1', float(loss1.detach()), cur_it)
                    writer.add_scalar('loss/loss2', float(loss2.detach()), cur_it)
                    if 'modal_selection' in Y:
                        writer.add_scalar(
                            'fusion/modal_selection_ratio',
                            float(Y['modal_selection'].mean()),
                            cur_it
                        )
                        valid_pixels = (
                            Y['modal_valid'].sum()
                            * Y['modal_selection'].shape[-2]
                            * Y['modal_selection'].shape[-1]
                        )
                        if valid_pixels.item() > 0:
                            writer.add_scalar(
                                'fusion/modal_selection_ratio_valid',
                                float(
                                    Y['modal_selection'].sum()
                                    / valid_pixels
                                ),
                                cur_it
                            )
                    for group_index, group in enumerate(optim.param_groups):
                        writer.add_scalar(
                            'learning_rate/group_{}'.format(group_index), group['lr'], cur_it
                        )
                if (writer is not None and tokens is not None and
                        (cur_it == 1 or cur_it % config['tb_image_interval'] == 0)):
                    log_token_alignment(
                        writer, images, tokens, flips, cur_it, config['tb_max_images']
                    )
                    if 'modal_selection' in Y:
                        selection = Y['modal_selection'][
                            :config['tb_max_images']
                        ].detach().float().cpu()
                        writer.add_image(
                            'fusion/modal_selection_white_is_modal',
                            make_grid(selection, nrow=1),
                            cur_it
                        )

            loss_count += loss.data
            adb_count += loss0
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            lrs = ','.join([format(param['lr'], ".2e") for param in optim.param_groups])
            if stage == 1:
                Bar.suffix = '{:4}/{:4} | loss: {:1.3f}, csd: {:1.3f}, btm: {:1.3f}, mse: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                    i, num_iter, float(loss_count / i), float(adb_count / i), float(ac_count / i), float(mse_count / i), lrs, time.time() - st)
            else:
                Bar.suffix = '{:4}/{:4} | loss: {:1.3f}, iou: {:1.3f}, bce: {:1.3f}, mse: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(i, num_iter, float(loss_count / i), float(adb_count / i), float(ac_count / i), float(mse_count / i), lrs, time.time() - st)
            bar.next()
            
            if epoch > 1 and stage > 1 and config['olr']:
                lamda = [0.2,0.6,0.2]
                for gt_path, image, pred, gt, flip in zip(gt_names, images, torch.sigmoid(Y['final'].detach()), gts, flips):
                    pred = F.interpolate(pred.unsqueeze(0), size=gt.size()[1:], mode='bilinear', align_corners=True)[0]
                    ref = par(image.unsqueeze(0),pred.unsqueeze(0)).squeeze(0)
                    ref = ref/ref.max()
                    if flip:
                        pred = pred.flip(2)
                        ref = ref.flip(2)
                        gt = gt.flip(2)
                    new_gt = (pred * lamda[0]).cpu().numpy().transpose(1, 2, 0)+(gt * lamda[1]).cpu().numpy().transpose(1, 2, 0)+(ref*lamda[2]).cpu().numpy().transpose(1, 2, 0)
                    new_gt = ((new_gt/new_gt.max())).astype(np.float32)
                    cv2.imwrite(gt_path, new_gt * 255)
                
        sche.step()
        bar.finish()
        #torch.cuda.empty_cache()
        i = len(train_loader)
        print('| loss: {:1.3f}, dfs: {:1.3f}, bac: {:1.3f}, mse: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                    float(loss_count / i), float(adb_count / i), float(ac_count / i), float(mse_count / i), lrs, time.time() - st))
        
        #if num_epoch-epoch<5:
        if epoch != config['epoch']:
            test_model(model, test_sets, config, epoch, mode='train')
        else:
            test_model(model, test_sets, config, epoch, mode='test')

    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()
