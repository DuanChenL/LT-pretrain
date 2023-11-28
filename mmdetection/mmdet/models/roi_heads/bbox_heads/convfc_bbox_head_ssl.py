import torch.nn as nn
import torch
import json
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS, build_loss
from .bbox_head import BBoxHead
from mmdet.models.losses import accuracy


def loss_function(predict, target):
    """
    损失函数，比较余弦相似度。归一化的欧氏距离等价于余弦相似度
    :param predict: online net输出的prediction
    :param target: target网络输出的projection
    :return: loss(损失)
    """
    return 2. - 2. * torch.cosine_similarity(predict, target, dim=-1)

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
        # loss_ssl = dict(
        #     type='ContrastiveLoss',
        #     loss_weight=0.1)
        loss_ssl = dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
        self.loss_ssl = build_loss(loss_ssl)

        self.m = 0.999
        K =8196
        dim = 1024
        self.K = K
        self.dim = dim
        self.register_buffer("queue_boxes", torch.randn(dim, K))
        self.queue_boxes = nn.functional.normalize(self.queue_boxes, dim=0)
        # self.register_buffer("queue_l", torch.randint(0, self.num_classes + 1, (K,)))
        self.register_buffer("queue_ptr_boxes", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_bg", torch.randn(dim, K))
        self.queue_boxes = nn.functional.normalize(self.queue_boxes, dim=0)
        # self.register_buffer("queue_l", torch.randint(0, self.num_classes + 1, (K,)))
        self.register_buffer("queue_ptr_bg", torch.zeros(1, dtype=torch.long))
        self.encoder_q = nn.Sequential(nn.Linear(1024, dim, bias=False), nn.BatchNorm1d(dim),
                                    nn.ReLU(True),
                                    nn.Linear(dim, dim))
        self.encoder_k = nn.Sequential(nn.Linear(1024, dim, bias=False), nn.BatchNorm1d(dim),
                                    nn.ReLU(True),
                                    nn.Linear(dim, dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
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
            # self.fc_cls_reweight = nn.Linear(256, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

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

    # SSL neg queue
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if self.K % batch_size == 0:
            ptr = int(self.queue_ptr_boxes)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.queue_boxes[:, ptr: ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr_boxes[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_bg(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if self.K % batch_size == 0:
            ptr = int(self.queue_ptr_bg)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.queue_bg[:, ptr: ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr_bg[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, y):
        # shared part
        logits = 0
        logits_down = 0
        cls_score_ss = 0
        logits_bg = 0
        if isinstance(y, list):
            x = y[0]
        else:
            x = y

        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        if len(y) == 4:
            x_2 = y[1]
            x_sim = y[2]
            x_sim_2 = y[3]

            if self.num_shared_convs > 0:
                for conv in self.shared_convs:
                    x_2 = conv(x_2)
                    x_sim = conv(x_sim)
                    x_sim_2 = conv(x_sim_2)

            if self.num_shared_fcs > 0:
                if self.with_avg_pool:
                    x_2 = self.avg_pool(x_2)
                    x_sim = self.avg_pool(x_sim)
                    x_sim_2 = self.avg_pool(x_sim_2)

                x_2 = x_2.flatten(1)
                x_sim = x_sim.flatten(1)
                x_sim_2 = x_sim_2.flatten(1)

                for fc in self.shared_fcs:
                    x_2 = self.relu(fc(x_2))
                    x_sim = self.relu(fc(x_sim))
                    x_sim_2 = self.relu(fc(x_sim_2))

            # SSL loss
            q = x_2
            q = self.encoder_q(q)
            q = nn.functional.normalize(q, dim=1)
            # total_entries = x.size(0)
            # bg = x[256:512].clone()
            # 创建一个索引列表，用于取出所需的数据
            # for index in range(512, total_entries, 512):
            #     bg = torch.cat((bg, x[index + 256:index + 512]), dim=0)
            with torch.no_grad():
                self._momentum_update_key_encoder()
                k = x_sim
                k_2 = x_sim_2
                k = self.encoder_k(k)
                k = torch.nn.functional.normalize(k, dim=1)
                k_2 = self.encoder_k(k_2)
                k_2 = torch.nn.functional.normalize(k_2, dim=1)
                # bg = self.encoder_k(bg)
                # bg = torch.nn.functional.normalize(bg, dim=1)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            # self._dequeue_and_enqueue_bg(bg)
            # l_neg_bg = torch.einsum("nc,ck->nk", [q, self.queue_bg.clone().detach()])
            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            l_pos_down = torch.einsum("nc,nc->n", [q, k_2]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue_boxes.clone().detach()])

            logits = torch.cat([l_pos, l_neg], dim=1) / 0.2
            logits_down = torch.cat([l_pos_down, l_neg], dim=1) / 0.2
            # logits_bg = torch.cat([l_pos, l_neg_bg], dim=1) / 0.07
            # logits_all = torch.cat([logits, logits_down], dim=0)

            self._dequeue_and_enqueue(k)
            self._dequeue_and_enqueue(k_2)

        # separate branches
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

        # cls_score = None
        # if self.with_cls:
        #     reweight = self.reweight_(x_cls)
        #     reweight = self.reweight_fc(reweight)
        #     reweight = reweight.view(-1, 256)
        #     cls_score = self.fc_cls_reweight(reweight)
        #     cls_score = cls_score.view(-1, self.num_classes + 1, self.num_classes + 1)
        #     cls_score = torch.diagonal(cls_score, dim1=-2, dim2=-1)
        #     cls_score = cls_score.view(-1, self.num_classes + 1)

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, [logits, logits_down, logits_bg]

    def loss(self,
             cls_score,
             bbox_pred,
             ssl_logits,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()

        # ssl_logits = F.softmax(ssl_logits, dim=1)
        # loss_ssl_ = -torch.log(ssl_logits[:, 0] / ssl_logits.sum(dim=1)).mean()
        # lam = 1.0
        # cls_score_ss, _ = torch.max(ssl_logits[-1], dim=1)
        # cls_score_ss = torch.where(cls_score_ss < 0, torch.tensor(0.9).to('cuda'), cls_score_ss)
        # cls_score_ss = torch.where(cls_score_ss > 1, torch.tensor(1.0).to('cuda'), cls_score_ss)
        # cls_score_ss = 1.08 - cls_score_ss
        # cls_score_ss = torch.unsqueeze(cls_score_ss, dim=-1)
        # ssl_weight = cls_score_ss
        # print(ssl_logits[0])
        # mse_pre = dict()
        # mse_ori = dict()
        # mse_pre['r'] = []
        # mse_pre['f'] = []
        # mse_pre['c'] = []
        # mse_ori['r'] = []
        # mse_ori['f'] = []
        # mse_ori['c'] = []
        # reconstructs = self.outputs
        # loss_mse_ = dict()
        # count_ind = 0
        # for i in range(len(gt_labels)):
        #     if len(reconstructs) == 0:
        #         reconstructs = self.outputs[i*128:i*128+len(gt_labels[i])]
        #     else:
        #         reconstructs = torch.cat((reconstructs, self.outputs[i*128:i*128+len(gt_labels[i])]), dim=0)
        # kl_divergence = -0.5 * torch.sum(1 + self.logvar - self.mean.pow(2) - self.logvar.exp())
        # loss_mse_['loss_mse'] = self.loss_mse(reconstructs, ori_gt_bboxes) + kl_divergence
        # if len(reconstructs) > len(ori_gt_bboxes):
        #     loss_mse_['loss_mse'] = self.loss_mse(reconstructs[:len(ori_gt_bboxes)], ori_gt_bboxes)
        # else:
        #     loss_mse_['loss_mse'] = self.loss_mse(reconstructs, ori_gt_bboxes[:len(reconstructs)])

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
        labels_ssl = torch.zeros(ssl_logits[0].shape[0], dtype=torch.long).cuda()
        # print(ssl_logits.shape)
        if labels_ssl.shape[0] == ssl_logits[0].shape[0] and labels_ssl.shape[0] == ssl_logits[1].shape[0]:
            loss_ssl_ = 0.5 * self.loss_ssl(
                ssl_logits[0],
                labels_ssl,
                ) + 0.5 * self.loss_ssl(
                ssl_logits[1],
                labels_ssl,
                )
            # loss_ssl_bg = self.loss_ssl(
            #     ssl_logits[2],
            #     labels_ssl
            # )
            loss_ssl_ = 0.05 * loss_ssl_
                        # + 0.05 * loss_ssl_bg
        if isinstance(loss_ssl_, dict):
            losses.update(loss_ssl_)
        else:
            losses['loss_ssl_bbox'] = loss_ssl_
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
            # if isinstance(loss_mse_, dict):
            #     losses.update(loss_mse_)
            # else:
            #     losses['loss_mse'] = loss_mse_
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