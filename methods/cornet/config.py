import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    # Default configure
    cfg_dict = {
        'optim': 'SGD',
        'schedule': 'StepLR',
        'lr': 0.005,
        'batch': 8,
        'ave_batch': 1,
        'epoch': 10,
        'step_size': '20,24',
        'gamma': 0.5,
        'clip_gradient': 0,
        'test_batch': 1
    }
    
    parser = base_config(cfg_dict)
    parser.set_defaults(stage=2)
    # Add custom params here
    parser.add_argument('--resdual', default=0.4, type=float)
    parser.add_argument(
        '--token_root',
        default='/home/zzx/sod/VRF_WSOD/TokenCut',
        type=str,
        help='Parent directory of the offline *_token_similarity_* folders'
    )
    parser.add_argument('--token_split', default='train', type=str)
    parser.add_argument('--token_dim', default=384, type=int)
    parser.add_argument('--tensorboard_dir', default='./runs', type=str)
    parser.add_argument('--tb_scalar_interval', default=10, type=int)
    parser.add_argument('--tb_image_interval', default=200, type=int)
    parser.add_argument('--tb_max_images', default=4, type=int)
    parser.add_argument('--rgb_aux_weight', default=0.2, type=float)
    parser.add_argument('--modal_aux_weight', default=0.2, type=float)
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    config['use_offline_token'] = True
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    # Use framework_factory's default two groups. Both encoder names contain
    # "encoder." and receive lr / 10; fusion and decoder use the base lr.
    config['lr_decay'] = 0.9
    
    return config, None
