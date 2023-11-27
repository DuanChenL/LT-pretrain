# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.models.builder import build_loss

# from .selective_search import selective_search
# import numpy as np
# from PIL import Image
import json


def bbox_flip(bboxes, img_shape, direction):
    assert bboxes.shape[-1] % 4 == 0
    flipped = bboxes.clone().detach()  # Use .clone() instead of .copy() for PyTorch tensors
    if direction == 'horizontal':
        w = img_shape[1]
        flipped[..., 0::4] = w - bboxes[..., 2::4]
        flipped[..., 2::4] = w - bboxes[..., 0::4]
    elif direction == 'vertical':
        h = img_shape[0]
        flipped[..., 1::4] = h - bboxes[..., 3::4]
        flipped[..., 3::4] = h - bboxes[..., 1::4]
    elif direction == 'diagonal':
        w = img_shape[1]
        h = img_shape[0]
        flipped[..., 0::4] = w - bboxes[..., 2::4]
        flipped[..., 1::4] = h - bboxes[..., 3::4]
        flipped[..., 2::4] = w - bboxes[..., 0::4]
        flipped[..., 3::4] = h - bboxes[..., 1::4]
    else:
        raise ValueError(f"Invalid flipping direction '{direction}'")

    return flipped.clone().detach()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.m = 0.999
        self.backbone = build_backbone(backbone)
        self.backbone_sim = build_backbone(backbone)

        K = 8196
        dim = 256
        self.K = K
        self.dim = dim
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        # self.register_buffer("queue_l", torch.randint(0, self.num_classes + 1, (K,)))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_layer2", torch.randn(dim, K))
        self.queue_layer2 = nn.functional.normalize(self.queue_layer2, dim=0)
        # self.register_buffer("queue_l", torch.randint(0, self.num_classes + 1, (K,)))
        self.register_buffer("queue_ptr_layer2", torch.zeros(1, dtype=torch.long))
        self.proj_q = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim),
                                    nn.ReLU(True),
                                    nn.Linear(dim, dim))
        self.proj_k = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim),
                                    nn.ReLU(True),
                                    nn.Linear(dim, dim))
        # loss_ssl = dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=False,
        #     loss_weight=1.0)
        # self.loss_ssl = build_loss(loss_ssl)

        loss_ssl = dict(
            type='ContrastiveLoss',
            loss_weight=0.1)
        self.loss_ssl = build_loss(loss_ssl)
        # self.decoder = nn.Sequential(nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
        #                              nn.ReLU(),
        #                              nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
        #                              nn.ReLU(),
        #                              nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        #                              nn.ReLU(),
        #                              nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
        #                              # nn.ReLU(),
        #                              # nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #                              nn.ReLU(),
        #                              nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        #                              # nn.Sigmoid()
        #                              )
        # loss_mse = dict(
        #     type='MSELoss',
        #     loss_weight=0.3)
        # self.loss_mse = build_loss(loss_mse)

        with open('/data1/PycharmProjects/dcl/long-tail-detection/oln_bbox.json', 'r') as file:
            ss_boxes = json.load(file)
        self.ss_boxes = ss_boxes

        if neck is not None:
            self.neck = build_neck(neck)
            self.neck_sim = build_neck(neck)

        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_sim.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.neck.parameters(), self.neck_sim.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if self.K % batch_size == 0:
            ptr = int(self.queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr: ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_sim.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.neck.parameters(), self.neck_sim.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img, return_feat=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        feat_last = x[-1]
        if self.with_neck:
            x = self.neck(x)
        if return_feat:
            return x, feat_last
        else:
            return x

    def extract_sim_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_sim(img)
        if self.with_neck:
            x = self.neck_sim(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        device = img[0].device
        # print(device)
        img_g = img[1]
        img_2 = img[2]
        # img_2 = kwargs['sim']['img']
        view2 = nn.functional.interpolate(img_2, scale_factor=0.5)
        select_proposal = list()
        for meta_info in img_metas[0]:
            img_name = meta_info['ori_filename'].split('/')[-1]
            boxes = [item[:4] for item in self.ss_boxes[img_name]]
            boxes = torch.tensor(boxes)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            indices = torch.randperm(len(boxes))[:8]
            # 从原始Tensor中挑选256个元素，并将它们转换为CUDA的Tensor
            boxes = boxes[indices].to(device)
            select_proposal.append(boxes)
            # print(select_proposal)
        selective_search_proposal = dict()
        selective_search_proposal['view0'] = select_proposal.copy()
        selective_search_proposal['view1'] = select_proposal.copy()
        selective_search_proposal['view2'] = select_proposal.copy()
        for i in range(len(select_proposal)):
            selective_search_proposal['view0'][i] = selective_search_proposal['view0'][i].float()
            scale = torch.tensor(img_metas[0][i]['scale_factor']).to(device)
            selective_search_proposal['view0'][i] = selective_search_proposal['view0'][i] * scale
            if img_metas[0][i]['flip']:
                selective_search_proposal['view0'][i] = bbox_flip(selective_search_proposal['view0'][i], img_metas[0][i]['img_shape'], img_metas[0][i]['flip_direction'])

            selective_search_proposal['view1'][i] = selective_search_proposal['view1'][i].float()
            scale = torch.tensor(img_metas[2][i]['scale_factor']).to(device)
            selective_search_proposal['view1'][i] = selective_search_proposal['view1'][i] * scale
            if img_metas[2][i]['flip']:
                selective_search_proposal['view1'][i] = bbox_flip(selective_search_proposal['view1'][i], img_metas[2][i]['img_shape'], img_metas[2][i]['flip_direction'])


            selective_search_proposal['view2'][i] = selective_search_proposal['view2'][i].float()
            selective_search_proposal['view2'][i] = selective_search_proposal['view1'][i] / 2
        selective_search_proposal['view0'] = tuple(selective_search_proposal['view0'])
        selective_search_proposal['view1'] = tuple(selective_search_proposal['view1'])
        selective_search_proposal['view2'] = tuple(selective_search_proposal['view2'])

        x, feat = self.extract_feat(img[0], return_feat=True)
        # output = self.decoder(feat)
        # loss_mse = self.loss_mse(output, img[0])

        x_g = torch.Tensor(x[-1].mean(-1).mean(-1))
        q = self.proj_q(x_g)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            x_g_sim = self.extract_sim_feat(img_g)
            x_bbox_sim = self.extract_sim_feat(img_2)
            view1 = x_bbox_sim
            view2 = self.extract_sim_feat(view2)
            x_sim_g_4 = torch.Tensor(x_g_sim[-1].mean(-1).mean(-1))
            k = x_sim_g_4
            k = self.proj_k(k)
            k = torch.nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        # l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits = torch.cat([l_pos, l_neg], dim=1) / 0.2
        # labels_ssl = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()
        # labels_ssl = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas[0],
                gt_bboxes[0],
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train([x, view1, view2], img_metas[0], proposal_list,
                                                 gt_bboxes[0], gt_labels[0],
                                                 gt_bboxes_ignore, gt_masks[0],
                                                 selective_search_proposal, img[0],
                                                 **kwargs)
        # roi_losses = self.roi_head.forward_train(x, img_metas[0], proposal_list,
        #                                          gt_bboxes[0], gt_labels[0],
        #                                          gt_bboxes_ignore, gt_masks[0],
        #                                          **kwargs)
        losses.update(roi_losses)

        # # loss_ssl_ = 0.1 * (torch.log(torch.exp(logits).sum(1) + 1e-5) - (logits * targets_con).sum(1)).mean(0)
        # if labels_ssl.shape[0] == logits.shape[0]:
        #     loss_ssl = self.loss_ssl(
        #         logits,
        #         labels_ssl)
        #     loss_ssl_ = 0.1 * loss_ssl
        loss_ssl_ = self.loss_ssl(q, k, self.queue.clone())
        if isinstance(loss_ssl_, dict):
            losses.update(loss_ssl_)
        else:
            losses['loss_ssl'] = loss_ssl_
        self._dequeue_and_enqueue(k)
        # if isinstance(loss_mse, dict):
        #     losses.update(loss_mse)
        # else:
        #     losses['loss_mse_global'] = loss_mse

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_sim_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
