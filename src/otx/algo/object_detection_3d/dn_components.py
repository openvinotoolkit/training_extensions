# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from utils import box_ops
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def prepare_for_dn(dn_args, tgt_weight, embedweight, batch_size, training, num_queries, num_classes, hidden_dim, label_enc):
    """
    The major difference from DN-DAB-DETR is that the author process pattern embedding pattern embedding in its detector
    forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, scalar, label_noise_scale, box_noise_scale, num_patterns
    :param tgt_weight: use learnbal tgt in dab deformable detr
    :param embedweight: positional anchor queries
    :param batch_size: bs
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn
    :return:
    """

    if training:
        targets, scalar, label_noise_scale, box_noise_scale, num_patterns = dn_args
    else:
        num_patterns = dn_args

    if num_patterns == 0:
        num_patterns = 1
    ###as groupdetr
    if training:
        indicator0 = torch.zeros([num_queries * num_patterns*11, 1]).cuda()
    else:
        indicator0 = torch.zeros([num_queries * num_patterns, 1]).cuda()
    ###
    # indicator0 = torch.zeros([num_queries * num_patterns, 1]).cuda()
    # sometimes the target is empty, add a zero part of label_enc to avoid unused parameters
    tgt = torch.cat([tgt_weight, indicator0], dim=1) + label_enc.weight[0][0]*torch.tensor(0).cuda()
    refpoint_emb = embedweight
    if training:
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]
        # you can uncomment this to use fix number of dn queries
        # if int(max(known_num))>0:
        #     scalar=scalar//int(max(known_num))

        # can be modified to selectively denosie some label or boxes; also known label prediction
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes_3d'] for t in targets])
        ## for 3d
        depths = torch.cat([t['depth'] for t in targets])
        size_3ds = torch.cat([t['size_3d'] for t in targets])
        target_heading_cls = torch.cat([t['heading_bin']for t in targets])
        target_heading_res = torch.cat([t['heading_res']for t in targets])
        ####
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # add noise
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_bboxs = boxes.repeat(scalar, 1)
        #for 3d
        known_depths = depths.repeat(scalar, 1)
        known_size_3ds = size_3ds.repeat(scalar, 1)
        known_target_heading_cls = target_heading_cls.repeat(scalar, 1)
        known_target_heading_res = target_heading_res.repeat(scalar, 1)
        ###
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # noise on the label
        if label_noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of bbox noise
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
     
            new_label=new_label.to(torch.int8)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        # noise on the box
        if box_noise_scale > 0:
            diff = torch.zeros_like(known_bbox_expand)
        
            # diff[:, :2] = known_bbox_expand[:, 2:] / 2
            diff[:, 0] = (known_bbox_expand[:, 2]+ known_bbox_expand[:,3])/ 2
            diff[:, 1] = (known_bbox_expand[:, 4]+ known_bbox_expand[:,5])/ 2
            diff[:, 2:] = known_bbox_expand[:, 2:]
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                           diff).cuda() * box_noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
 
        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        # add dn part indicator
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 6).cuda()
       
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
        input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
     
        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
        # for groupdetr
        tgt_size = pad_size + num_queries * num_patterns
        ###
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs, known_size_3ds, known_depths, known_target_heading_cls, known_target_heading_res),
            'know_idx': know_idx,
            'pad_size': pad_size
        }
    else:  # no dn for inference
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

    return input_query_label, input_query_bbox, attn_mask, mask_dict


def dn_post_process(outputs_class, outputs_coord, outputs_3d_dim,outputs_depth,outputs_angle,mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        output_known_3d_dim = outputs_3d_dim[:, :, :mask_dict['pad_size'], :]
        output_known_depth = outputs_depth[:, :, :mask_dict['pad_size'], :]
        output_known_angle = outputs_angle[:, :, :mask_dict['pad_size'], :]

        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        outputs_3d_dim = outputs_3d_dim[:, :, mask_dict['pad_size']:, :]
        outputs_depth = outputs_depth[:, :, mask_dict['pad_size']:, :]
        outputs_angle = outputs_angle[:, :, mask_dict['pad_size']:, :]
        mask_dict['output_known_lbs_bboxes']=(output_known_class,output_known_coord,output_known_3d_dim,output_known_depth,output_known_angle)
    return outputs_class, outputs_coord, outputs_3d_dim, outputs_depth, outputs_angle


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    Returns:

    """

    output_known_class, output_known_coord, output_known_3d_dim,output_known_depth,output_known_angle = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs, known_size_3ds, known_depths, known_target_heading_cls, known_target_heading_res = mask_dict['known_lbs_bboxes']
    map_known_indice = mask_dict['map_known_indice']

    known_indice = mask_dict['known_indice']

    batch_idx = mask_dict['batch_idx']
    bid = batch_idx[known_indice]
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_3d_dim = output_known_3d_dim.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_depth = output_known_depth.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_angle = output_known_angle.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
    num_tgt = known_indice.numel()
    return known_labels, known_bboxs, known_size_3ds, known_depths, known_target_heading_cls, known_target_heading_res, output_known_class, output_known_coord, output_known_3d_dim,output_known_depth,output_known_angle, num_tgt


def tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt,):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    if len(tgt_boxes) == 0:
        return {
            'tgt_loss_bbox': torch.as_tensor(0.).to('cuda'),
            'tgt_loss_giou': torch.as_tensor(0.).to('cuda'),
        }

    loss_bbox = F.l1_loss(src_boxes[:,2:6], tgt_boxes[:,2:6], reduction='none')

    losses = {}
    losses['tgt_loss_bbox'] = loss_bbox.sum() / num_tgt

    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcylrtb_to_xyxy(src_boxes),
        box_ops.box_cxcylrtb_to_xyxy(tgt_boxes)))
    losses['tgt_loss_giou'] = loss_giou.sum() / num_tgt
    return losses

def tgt_loss_3dcenter(src_boxes, tgt_boxes, num_tgt):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """

    loss_3dcenter = F.l1_loss(src_boxes[:,0:2], tgt_boxes[:,0:2], reduction='none')
    losses = {}
    losses['tgt_loss_center'] = loss_3dcenter.sum() / num_tgt
    return losses

def tgt_loss_depths(src_depth, tgt_depth, num_tgt):  

    depth_input, depth_log_variance = src_depth[:, 0], src_depth[:, 1] 
    depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - tgt_depth) + depth_log_variance  
    losses = {}
    losses['tgt_loss_depth'] = depth_loss.sum() / num_tgt 
    return losses  
    
def tgt_loss_dims(src_dim, tgt_dim, num_tgt):  

    dimension = tgt_dim.clone().detach()
    dim_loss = torch.abs(src_dim - tgt_dim)
    dim_loss /= dimension
    with torch.no_grad():
        compensation_weight = F.l1_loss(src_dim, tgt_dim) / dim_loss.mean()
    dim_loss *= compensation_weight
    losses = {}
    losses['tgt_loss_dim'] = dim_loss.sum() / num_tgt
    return losses

def tgt_loss_angles(heading_input, target_heading_cls, target_heading_res, num_tgt):  

    heading_input = heading_input.view(-1, 24)
    heading_target_cls = target_heading_cls.view(-1).long()
    heading_target_res = target_heading_res.view(-1)

    # classification loss
    heading_input_cls = heading_input[:, 0:12]
    cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

    # regression loss
    heading_input_res = heading_input[:, 12:24]
    cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
    heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
    reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
    
    angle_loss = cls_loss + reg_loss
    losses = {}
    losses['tgt_loss_angle'] = angle_loss.sum() / num_tgt 
    return losses

def tgt_loss_labels(src_logits_, tgt_labels_, num_tgt, focal_alpha, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_ce': torch.as_tensor(0.).to('cuda'),
            'tgt_class_error': torch.as_tensor(0.).to('cuda'),
        }

    src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0).to(torch.int64)

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:, :, :-1]
    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=focal_alpha, gamma=2) * src_logits.shape[1]

    losses = {'tgt_loss_ce': loss_ce}

    losses['tgt_class_error'] = 100 - accuracy(src_logits_, tgt_labels_)[0]
    return losses


def compute_dn_loss(mask_dict, training, aux_num, focal_alpha):
    """
       compute dn loss in criterion
       Args:
           mask_dict: a dict for dn information
           training: training or inference flag
           aux_num: aux loss number
           focal_alpha:  for focal loss
       """
    losses = {}
    if training and 'output_known_lbs_bboxes' in mask_dict:
        known_labels, known_bboxs, known_size_3ds, known_depths, known_target_heading_cls, known_target_heading_res,\
        output_known_class, output_known_coord, output_known_3d_dim,output_known_depth,output_known_angle, num_tgt = prepare_for_loss(mask_dict)
     
        losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, focal_alpha))
        losses.update(tgt_loss_boxes(output_known_coord[-1], known_bboxs, num_tgt))
        losses.update(tgt_loss_3dcenter(output_known_coord[-1], known_bboxs, num_tgt))
        # losses.update(tgt_loss_depths(output_known_depth[-1], known_depths, num_tgt))
        #losses.update(tgt_loss_dims(output_known_3d_dim[-1], known_size_3ds, num_tgt))
        losses.update(tgt_loss_angles(output_known_angle[-1], known_target_heading_cls, known_target_heading_res, num_tgt))
    else:
        losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_center'] = torch.as_tensor(0.).to('cuda')
        # losses['tgt_loss_depth'] = torch.as_tensor(0.).to('cuda')
        #losses['tgt_loss_dim'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_angle'] = torch.as_tensor(0.).to('cuda')
        

    if aux_num:
        for i in range(aux_num):
            # dn aux loss
            if training and 'output_known_lbs_bboxes' in mask_dict:
                l_dict = tgt_loss_labels(output_known_class[i], known_labels, num_tgt, focal_alpha)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_boxes(output_known_coord[i], known_bboxs, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_3dcenter(output_known_coord[i], known_bboxs, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_angles(output_known_angle[-1], known_target_heading_cls, known_target_heading_res, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_center'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_angle'] = torch.as_tensor(0.).to('cuda')
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses