import torch

from ..bbox_utils import decode, nms
from torch.autograd import Function


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    @staticmethod

    def forward(ctx, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        ctx.num_classes = 2
        ctx.top_k = 750
        ctx.nms_thresh = 0.3
        ctx.conf_thresh = 0.05
        ctx.variance = [0.1, 0.2]
        ctx.nms_top_k = 5000

        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(
            num, num_priors, ctx.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors,
                                       4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4),
                               batch_priors, ctx.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, ctx.num_classes, ctx.top_k, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, ctx.num_classes):
                c_mask = conf_scores[cl].gt(ctx.conf_thresh)
                scores = conf_scores[cl][c_mask]
                
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(
                    boxes_, scores, ctx.nms_thresh, ctx.nms_top_k)
                count = count if count < ctx.top_k else ctx.top_k

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes_[ids[:count]]), 1)

        return output
