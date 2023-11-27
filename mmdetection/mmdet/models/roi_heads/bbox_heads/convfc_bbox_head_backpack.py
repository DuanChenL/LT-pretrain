import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS, build_loss
from .bbox_head import BBoxHead
from .self_atten import RelationReasoning
from mmdet.models.losses import accuracy
import json

@HEADS.register_module()
class SSLConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(SSLConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        # self.sense_vector = nn.Linear(1, 5)
        # self.RelationReasoning = RelationReasoning(num_classes=self.num_classes)
        self.decoder = nn.Sequential(
                                     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                                     # nn.Sigmoid()
                                     )
        loss_mse = dict(
            type='MSELoss',
            loss_weight=1.0)
        self.loss_mse = build_loss(loss_mse)
        with open('/data1/PycharmProjects/dcl/long-tail-detection/baseline/cat2fre.json', 'r') as file:
        # with open('/data/duancl/ssl_LT_detection/baseline/cat2fre.json', 'r') as file:
            cat2fre = json.load(file)
        self.cat2fre = cat2fre

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(SSLConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self.outputs = self.decoder(x)
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls = x
        x_reg = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             gt_labels,
             ori_gt_bboxes,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        mse_pre = dict()
        mse_ori = dict()
        mse_pre['r'] = []
        mse_pre['f'] = []
        mse_pre['c'] = []
        mse_ori['r'] = []
        mse_ori['f'] = []
        mse_ori['c'] = []
        reconstructs = []
        loss_mse_ = dict()
        count_ind = 0
        for i in range(len(gt_labels)):
            if len(reconstructs) == 0:
                reconstructs = self.outputs[i * 512:i * 512 + len(gt_labels[i])]
            else:
                reconstructs = torch.cat((reconstructs, self.outputs[i * 512:i * 512 + len(gt_labels[i])]), dim=0)
            # reconstruct.append(self.outputs[i * 512:i * 512 + len(gt_labels[i])])
        for i in range(len(gt_labels)):
            for j in range(len(gt_labels[i])):
                if j > 64:
                    break
                if self.cat2fre[str((gt_labels[i][j] + 1).cpu().item())] == 'f':
                    mse_pre['f'].append(reconstructs[count_ind + j])
                    mse_ori['f'].append((ori_gt_bboxes[count_ind + j]))
                elif self.cat2fre[str((gt_labels[i][j] + 1).cpu().item())] == 'c':
                    mse_pre['c'].append(reconstructs[count_ind + j])
                    mse_ori['c'].append((ori_gt_bboxes[count_ind + j]))
                else:
                    mse_pre['r'].append(reconstructs[count_ind + j])
                    mse_ori['r'].append((ori_gt_bboxes[count_ind + j]))
            count_ind += len(gt_labels[i])
        # reconstructs = torch.cat(reconstruct, dim=0)
        if len(reconstructs) > len(ori_gt_bboxes):
            loss_mse_['loss_mse'] = self.loss_mse(reconstructs[:len(ori_gt_bboxes)], ori_gt_bboxes)
        else:
            loss_mse_['loss_mse'] = self.loss_mse(reconstructs, ori_gt_bboxes[:len(reconstructs)])
        loss_mse_['r_mse'] = torch.tensor(0.0).cuda()
        loss_mse_['c_mse'] = torch.tensor(0.0).cuda()
        loss_mse_['f_mse'] = torch.tensor(0.0).cuda()
        if len(mse_pre['r']) != 0:
            mse_pre['r'] = torch.stack(mse_pre['r'])
            mse_ori['r'] = torch.stack(mse_ori['r'])
            loss_mse_['r_mse'] = self.loss_mse(mse_pre['r'], mse_ori['r'])
        if len(mse_pre['c']) != 0:
            mse_pre['c'] = torch.stack(mse_pre['c'])
            mse_ori['c'] = torch.stack(mse_ori['c'])
            loss_mse_['c_mse'] = self.loss_mse(mse_pre['c'], mse_ori['c'])
        if len(mse_pre['f']) != 0:
            mse_pre['f'] = torch.stack(mse_pre['f'])
            mse_ori['f'] = torch.stack(mse_ori['f'])
            loss_mse_['f_mse'] = self.loss_mse(mse_pre['f'], mse_ori['f'])

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                # loss_cls_ = 1.0 * loss_cls_ + 0.5 * self.reloss(cls_score, labels)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if isinstance(loss_mse_, dict):
            losses.update(loss_mse_)
        else:
            losses['loss_mse'] = loss_mse_
        return losses


@HEADS.register_module()
class SSLShared2FCBBoxHead(SSLConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(SSLShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class SSLShared4Conv1FCBBoxHead(SSLConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(SSLShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)