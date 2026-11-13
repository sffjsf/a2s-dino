import copy

import torch
from torch import nn
from torch.nn import functional as F


INTERPOLATION_MODE = 'bilinear'


def up_conv(cin, cout, up=True):
    yield nn.Conv2d(cin, cout, 3, padding=1)
    yield nn.GroupNorm(cout // 2, cout)
    yield nn.ReLU(inplace=True)
    if up:
        yield nn.Upsample(
            scale_factor=2,
            mode=INTERPOLATION_MODE,
            align_corners=False,
        )


def local_conv(cin, cout):
    yield nn.Conv2d(cin, cout * 2, 3, padding=1)
    yield nn.GroupNorm(cout, cout * 2)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(
        scale_factor=2,
        mode=INTERPOLATION_MODE,
        align_corners=False,
    )
    yield nn.Conv2d(cout * 2, cout, 3, padding=1)
    yield nn.GroupNorm(cout // 2, cout)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(
        scale_factor=2,
        mode=INTERPOLATION_MODE,
        align_corners=False,
    )


class InfoBlock(nn.Module):
    def __init__(self, channels):
        super(InfoBlock, self).__init__()
        self.gconv = nn.Sequential(
            *list(up_conv(channels, channels, False))
        )
        self.res_conv1 = nn.Conv2d(
            channels, channels, 3, padding=1
        )
        self.res_conv2 = nn.Conv2d(
            channels, channels, 3, padding=1
        )
        self.fuse = nn.Conv2d(
            channels * 3, channels, 3, padding=1
        )

    def forward(self, xs, global_feature):
        target_size = xs[0].shape[-2:]
        global_feature = F.interpolate(
            self.gconv(global_feature),
            size=target_size,
            mode=INTERPOLATION_MODE,
            align_corners=False,
        )

        feature1 = xs[0]
        residual1 = torch.sigmoid(
            self.res_conv1(feature1 - global_feature)
        )
        feature2 = F.interpolate(
            xs[1],
            size=target_size,
            mode=INTERPOLATION_MODE,
            align_corners=False,
        )
        residual2 = torch.sigmoid(
            self.res_conv2(feature2 - global_feature)
        )

        output = self.fuse(torch.cat(
            (
                feature1 * residual1,
                feature2 * residual2,
                global_feature,
            ),
            dim=1,
        ))
        return output


class BranchDecoder(nn.Module):
    """One modality-specific CORNet Local/Region decoder."""

    def __init__(self, feat):
        super(BranchDecoder, self).__init__()
        channels = feat[0]
        self.adapters = nn.ModuleList([
            nn.Sequential(
                *list(up_conv(in_channels, channels, False))
            )
            for in_channels in feat
        ])
        self.region = InfoBlock(channels)
        self.local = InfoBlock(channels)
        self.global_to_local = nn.Sequential(
            *list(local_conv(channels, channels))
        )

    def forward(self, features):
        features = [
            adapter(feature)
            for adapter, feature in zip(self.adapters, features)
        ]

        global_feature = features[4]
        region = self.region(features[2:4], global_feature)
        local_global = self.global_to_local(global_feature)
        local = self.local(features[0:2], local_global)
        return local, region


class TokenRefine(nn.Module):
    """Use an RGB-derived token to refine one RGB decoder feature."""

    def __init__(self, token_dim, channels):
        super(TokenRefine, self).__init__()
        self.token_adapter = nn.Sequential(
            nn.Conv2d(token_dim, channels, 1, bias=False),
            nn.GroupNorm(channels // 2, channels),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels * 2, channels, 3, padding=1, bias=False
            ),
            nn.GroupNorm(channels // 2, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

        # Initially keep the RGB decoder unchanged and learn token
        # corrections gradually.
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, feature, token):
        token = F.interpolate(
            token,
            size=feature.shape[-2:],
            mode=INTERPOLATION_MODE,
            align_corners=False,
        )
        token = self.token_adapter(token)
        correction = self.refine(
            torch.cat((feature, token), dim=1)
        )
        return feature + correction


class PredictionHead(nn.Module):
    def forward(self, local, region, output_size):
        region = F.interpolate(
            region,
            size=local.shape[-2:],
            mode=INTERPOLATION_MODE,
            align_corners=False,
        )
        prediction = torch.sum(
            local * region, dim=1, keepdim=True
        )
        return F.interpolate(
            prediction,
            size=output_size,
            mode=INTERPOLATION_MODE,
            align_corners=False,
        )


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        self.only_rgb = config.get('only_rgb', False)
        channels = feat[0]
        token_dim = config.get('token_dim', 384)

        self.encoder = encoder
        self.rgb_decoder = BranchDecoder(feat)
        self.local_token_refine = TokenRefine(
            token_dim, channels
        )
        self.region_token_refine = TokenRefine(
            token_dim, channels
        )

        if self.only_rgb:
            self.modal_encoder = None
            self.modal_decoder = None
        else:
            self.modal_encoder = copy.deepcopy(encoder)
            # Keep the two decoders trainable independently, but start
            # from aligned channel semantics.
            self.modal_decoder = copy.deepcopy(self.rgb_decoder)

        self.prediction_head = PredictionHead()

        # Auxiliary images are returned in [0, 1], while RGB has already
        # been normalized by data.py.
        self.register_buffer(
            'modal_mean',
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1),
        )
        self.register_buffer(
            'modal_std',
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1),
        )

    def forward(self, inputs, phase='test'):
        image, modal, token, modal_type = self._unpack_inputs(
            inputs
        )
        output_size = image.shape[-2:]

        rgb_features = self.encoder(image)
        rgb_local, rgb_region = self.rgb_decoder(rgb_features)
        rgb_local = self.local_token_refine(rgb_local, token)
        rgb_region = self.region_token_refine(rgb_region, token)
        rgb_prediction = self.prediction_head(
            rgb_local, rgb_region, output_size
        )

        if self.only_rgb:
            return {
                'sal': [rgb_prediction],
                'final': rgb_prediction,
                'rgb': rgb_prediction,
            }

        modal_type = modal_type.to(
            device=image.device, dtype=torch.long
        ).view(-1)
        modal_valid = (
            modal_type.eq(1) | modal_type.eq(2)
        )
        modal_prediction = rgb_prediction.detach().clone()

        # Depth (type 1) and Thermal (type 2) share this exact
        # Encoder/Decoder. Pure RGB samples never execute the branch.
        if modal_valid.any():
            valid_indices = modal_valid.nonzero(
                as_tuple=False
            ).flatten()
            valid_modal = modal.index_select(
                0, valid_indices
            )
            valid_modal = (
                valid_modal - self.modal_mean
            ) / self.modal_std
            modal_features = self.modal_encoder(valid_modal)
            modal_local, modal_region = self.modal_decoder(
                modal_features
            )
            valid_modal_prediction = self.prediction_head(
                modal_local, modal_region, output_size
            )
            modal_prediction = modal_prediction.index_copy(
                0,
                valid_indices,
                valid_modal_prediction,
            )

        fused_prediction, modal_selection = (
            self.hard_confidence_fusion(
                rgb_prediction,
                modal_prediction,
                modal_valid,
            )
        )

        return {
            'sal': [
                fused_prediction,
                rgb_prediction,
                modal_prediction,
            ],
            'final': fused_prediction,
            'rgb': rgb_prediction,
            'modal': modal_prediction,
            'modal_valid': modal_valid.float(),
            'modal_type': modal_type,
            'modal_selection': modal_selection,
        }

    @staticmethod
    def hard_confidence_fusion(
            rgb_prediction, modal_prediction, modal_valid):
        """Select the more confident branch independently per pixel."""
        with torch.no_grad():
            rgb_confidence = torch.abs(
                torch.sigmoid(rgb_prediction) - 0.5
            )
            modal_confidence = torch.abs(
                torch.sigmoid(modal_prediction) - 0.5
            )
            modal_selection = (
                modal_confidence > rgb_confidence
            )
            modal_selection = modal_selection & modal_valid.view(
                -1, 1, 1, 1
            )

        fused_prediction = torch.where(
            modal_selection,
            modal_prediction,
            rgb_prediction,
        )
        return fused_prediction, modal_selection.float()

    def _unpack_inputs(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError(
                'CORNet input must be a list or tuple.'
            )

        if self.only_rgb:
            if len(inputs) != 2:
                raise ValueError(
                    'RGB-only CORNet expects '
                    '[rgb, offline_token].'
                )
            image, token = inputs
            modal = None
            modal_type = None
        else:
            if len(inputs) not in (3, 4):
                raise ValueError(
                    'Multimodal CORNet expects '
                    '[rgb, modal, offline_token, modal_type].'
                )
            image, modal, token = inputs[:3]
            if len(inputs) == 4:
                modal_type = inputs[3]
            else:
                # Backward-compatible single-modality call: every
                # sample has a valid auxiliary modality.
                modal_type = torch.ones(
                    image.shape[0],
                    device=image.device,
                    dtype=torch.long,
                )

        if token is None:
            raise ValueError(
                'CORNet requires a non-empty offline token map.'
            )
        return image, modal, token, modal_type
