import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    # Default configure
    cfg_dict = {
        'optim': 'SGD',
        'schedule': 'StepLR',
        'lr': 0.1,
        'batch': 8,
        'ave_batch': 1,
        'epoch': 20,
        'step_size': '12,16',
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1,
    }
    
    parser = base_config(cfg_dict)
    # Add custom params here
    parser.add_argument('--ac', default=0.05, type=float)
    parser.add_argument('--rgb', default=200, type=float)
    parser.add_argument('--lrw', type=str, default='1,0.05,1')
    parser.add_argument('--temp', default='temp1', help='Job name')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--token_root', default='/home/zzx/sod/VRF_WSOD/TokenCut', type=str)
    parser.add_argument('--token_split', default='train', type=str)
    parser.add_argument('--token_dim', default=384, type=int)
    parser.add_argument('--tensorboard_dir', default='./runs', type=str)
    parser.add_argument('--tb_scalar_interval', default=10, type=int)
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    config['use_stage1_dino_prior'] = config['stage'] == 1
    config['use_offline_token'] = config['stage'] == 1
    config['use_offline_token_test'] = False
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    # Config post-process
    if config['finetune']:
        config['params'] = [['encoder', 0], ['decoder', 0], ['adapter', config['lr']]]
        if config['weight'] == '':
            config['weight'] = config['load'] + 'a2s/base.pth'
        config['pcl'] = False
    else:
        config['params'] = [['encoder', 0], ['decoder', config['lr']]]
    config['lr_decay'] = 0.9
    
    return config, None
