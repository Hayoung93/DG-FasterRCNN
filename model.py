from types import MethodType
from collections import OrderedDict
from typing import Optional, List, Dict, Tuple, Union
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
import clip


# For training augmentation parameters
class AugTrainer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Backbone CNN
        if cfg.model.backbone == "resnet101":
            self.backbone = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
            # self.backbone = torch.nn.Sequential(*list(resnet101.children())[:-2])
        else:
            raise Exception("Not suppoerted backbone")
        def forward_layer3(_self, x):
            x = _self.maxpool(_self.relu(_self.bn1(_self.conv1(x))))
            x = _self.layer1(x)
            x = _self.layer2(x)
            x = _self.layer3(x)
            return x
        self.backbone._forward_impl = MethodType(forward_layer3, self.backbone)
        self.backbone.out_channels = list(list(eval("self.backbone.{}.children()".format(cfg.model.feat_layer)))[-1].children())[-2].num_features
        for n, p in self.backbone.named_parameters():
            p.requires_grad = False
        # Anchor generator
        # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        # aspect_ratios = ((0.5, 1.0, 2.0),)
        # self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        # RPN
        # featmap_names=['0'] for type: tensors - features from end of layer 3 will be passed
        # self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
        # Faster RCNN
        # self.model = FasterRCNN(self.backbone, num_classes=cfg.data.num_classes, rpn_anchor_generator=self.anchor_generator, box_roi_pool=self.roi_pooler)
        # rpn_head = torchvision.models.detection.rpn.RPNHead(self.backbone.out_channels, self.anchor_generator.num_anchors_per_location()[0])
        # rpn_pre_nms_top_n = dict(training=cfg.model.rpn.rpn_pre_nms_top_n_train, testing=cfg.model.rpn.rpn_pre_nms_top_n_test)
        # rpn_post_nms_top_n = dict(training=cfg.model.rpn.rpn_post_nms_top_n_train, testing=cfg.model.rpn.rpn_post_nms_top_n_test)
        # self.rpn = torchvision.models.detection.rpn.RegionProposalNetwork(self.anchor_generator, rpn_head, cfg.model.rpn.rpn_fg_iou_thresh, cfg.model.rpn.rpn_bg_iou_thresh, 
        #     cfg.model.rpn.rpn_batch_size_per_image, cfg.model.rpn.rpn_positive_fraction, rpn_pre_nms_top_n, rpn_post_nms_top_n, cfg.model.rpn.rpn_nms_thresh, score_thresh=cfg.model.rpn.rpn_score_thresh)
        # box_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        # resolution = box_roi_pool.output_size[0]
        # representation_size = 1024
        # box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(self.backbone.out_channels * resolution ** 2, representation_size)
        # box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(representation_size, cfg.data.num_classes)
        # self.roi_heads = torchvision.models.detection.roi_heads.RoIHeads(box_roi_pool, box_head, box_predictor, cfg.model.box.box_fg_iou_thresh, cfg.model.box.box_bg_iou_thresh,
        #     cfg.model.box.box_batch_size_per_image, cfg.model.box.box_positive_fraction, None, cfg.model.box.box_score_thresh, cfg.model.box.box_nms_thresh, cfg.model.box.box_detections_per_img)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(cfg.transform.min_size, cfg.transform.max_size, image_mean, image_std)
        # CLIP
        self.clip, self.preprocessor = clip.load('RN101')
        for n, p in self.clip.named_parameters():
            p.requires_grad = False
        # Domain offsets
        prompt_source = "an image taken during a day"
        self.prompts = {}
        domain_keys = []
        for weather in cfg.domain.weather_domains:
            for timezone in cfg.domain.time_domains:
                dkey = "{}-{}".format(weather, timezone)
                domain_keys.append(dkey)
                self.prompts.update({dkey: "an image taken on a {} {}".format(weather, timezone)})
        token_source = clip.tokenize(prompt_source)
        self.domain_tk = dict([(k, clip.tokenize(t)) for k, t in self.prompts.items()])
        with torch.no_grad():
            q_s = self.clip.encode_text(token_source.cuda())
            self.offsets = {}
            for k, v in self.domain_tk.items():
                with torch.no_grad():
                    q_t = self.clip.encode_text(v.cuda())
                    q_t = q_t / q_t.norm(dim=-1, keepdim=True)
                    text_off = q_t - q_s
                    text_off = text_off / text_off.norm(dim=-1, keepdim=True)
                    self.offsets[k] = text_off
        # Learnable parameters
        self.style_params, self.semantic_aug = nn.ParameterDict({}), nn.ParameterDict({})
        for wdomain in cfg.domain.weather_domains:
            for tdomain in cfg.domain.time_domains:
                # Style augmentation parameters
                self.style_params["{}-{}-mean".format(wdomain, tdomain)] = nn.Parameter(torch.zeros(cfg.domain.aug_dim))
                self.style_params["{}-{}-std".format(wdomain, tdomain)] = nn.Parameter(torch.ones(cfg.domain.aug_dim))
                # Addition augmentation parameters
                self.semantic_aug["{}-{}".format(wdomain, tdomain)] = nn.Parameter(torch.zeros(cfg.domain.aug_dim))
    
    def forward(self, images: Union[torch.Tensor, List[torch.Tensor]], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        losses = {}
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )
        # get outputs from the end of layer 3
        with torch.no_grad():
            features = self.backbone(images.tensors)
        # compute z: original feature
        with torch.no_grad():
            z = self.backbone.avgpool(self.backbone.layer4(features))
            z = z / z.norm(dim=1, keepdim=True)
        # compute z*: z + offset
        loss_cos, loss_l1 = 0.0, 0.0
        for domain, offset in self.offsets.items():
            with torch.no_grad():
                z_star = z + torch.nn.functional.interpolate(offset.unsqueeze(0), z.shape[1], mode="linear").permute(0, 2, 1)[..., None]
                z_star = z_star / z_star.norm(dim=1, keepdim=True)
            # compute z_bar: features that augmented with learnable parameters
            learnable_mean, learnable_std = self.style_params[domain + "-mean"], self.style_params[domain + "-std"]
            learnable_aug = self.semantic_aug[domain]
            feat_aug = features * learnable_std[None, :, None, None] + learnable_mean[None, :, None, None]
            feat_aug = feat_aug + learnable_aug[None, :, None, None]
            z_bar = self.backbone.avgpool(self.backbone.layer4(feat_aug))
            z_bar = z_bar / z_bar.norm(dim=1, keepdim=True)
            cos_dist = 1 - z_star.squeeze(-1).permute(0, 2, 1).bmm(z_bar.squeeze(-1))
            _loss_cos = cos_dist.mean()
            _loss_l1 = torch.nn.functional.l1_loss(z_bar, z)
            loss_cos += _loss_cos
            loss_l1 += _loss_l1
        losses["loss_cos"] = loss_cos
        losses["loss_l1"] = loss_l1
        # if isinstance(features, torch.Tensor):
        #     features = OrderedDict([("0", features)])
        # proposals, proposal_losses = self.rpn(images, features, targets)
        # detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        # losses.update(detector_losses)
        # losses.update(proposal_losses)

        return losses


class FasterRCNN_noFPN(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Backbone CNN
        if cfg.model.backbone == "resnet101":
            self.backbone = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
            # self.backbone = torch.nn.Sequential(*list(resnet101.children())[:-2])
        else:
            raise Exception("Not suppoerted backbone")
        def forward_layer3(_self, x):
            x = _self.maxpool(_self.relu(_self.bn1(_self.conv1(x))))
            x = _self.layer1(x)
            x = _self.layer2(x)
            x = _self.layer3(x)
            return x
        self.backbone._forward_impl = MethodType(forward_layer3, self.backbone)
        self.backbone.out_channels = list(list(eval("self.backbone.{}.children()".format(cfg.model.feat_layer)))[-1].children())[-2].num_features
        # Anchor generator
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),)
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        # RPN
        # featmap_names=['0'] for type: tensors - features from end of layer 3 will be passed
        # self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
        # Faster RCNN
        # self.model = FasterRCNN(self.backbone, num_classes=cfg.data.num_classes, rpn_anchor_generator=self.anchor_generator, box_roi_pool=self.roi_pooler)
        rpn_head = torchvision.models.detection.rpn.RPNHead(self.backbone.out_channels, self.anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=cfg.model.rpn.rpn_pre_nms_top_n_train, testing=cfg.model.rpn.rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=cfg.model.rpn.rpn_post_nms_top_n_train, testing=cfg.model.rpn.rpn_post_nms_top_n_test)
        self.rpn = torchvision.models.detection.rpn.RegionProposalNetwork(self.anchor_generator, rpn_head, cfg.model.rpn.rpn_fg_iou_thresh, cfg.model.rpn.rpn_bg_iou_thresh, 
            cfg.model.rpn.rpn_batch_size_per_image, cfg.model.rpn.rpn_positive_fraction, rpn_pre_nms_top_n, rpn_post_nms_top_n, cfg.model.rpn.rpn_nms_thresh, score_thresh=cfg.model.rpn.rpn_score_thresh)
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(self.backbone.out_channels * resolution ** 2, representation_size)
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(representation_size, cfg.data.num_classes)
        self.roi_heads = torchvision.models.detection.roi_heads.RoIHeads(box_roi_pool, box_head, box_predictor, cfg.model.box.box_fg_iou_thresh, cfg.model.box.box_bg_iou_thresh,
            cfg.model.box.box_batch_size_per_image, cfg.model.box.box_positive_fraction, None, cfg.model.box.box_score_thresh, cfg.model.box.box_nms_thresh, cfg.model.box.box_detections_per_img)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(cfg.transform.min_size, cfg.transform.max_size, image_mean, image_std)
        """
        # CLIP
        self.clip, self.preprocessor = clip.load('RN101')
        for n, p in self.clip.named_parameters():
            p.requires_grad = False
        # Domain offsets
        prompt_source = "an image taken during a day"
        self.prompts = {}
        domain_keys = []
        for weather in cfg.domain.weather_domains:
            for timezone in cfg.domain.time_domains:
                dkey = "{}-{}".format(weather, timezone)
                domain_keys.append(dkey)
                self.prompts.update({dkey: "an image taken on a {} {}".format(weather, timezone)})
        token_source = clip.tokenize(prompt_source)
        self.domain_tk = dict([(k, clip.tokenize(t)) for k, t in self.prompts.items()])
        with torch.no_grad():
            q_s = self.clip.encode_text(token_source.cuda())
            self.offsets = {}
            for k, v in self.domain_tk.items():
                with torch.no_grad():
                    q_t = self.clip.encode_text(v.cuda())
                    q_t = q_t / q_t.norm(dim=-1, keepdim=True)
                    text_off = q_t - q_s
                    text_off = text_off / text_off.norm(dim=-1, keepdim=True)
                    self.offsets[k] = text_off
        """
        # Learnable parameters
        self.style_params, self.semantic_aug = nn.ParameterDict({}), nn.ParameterDict({})
        for wdomain in cfg.domain.weather_domains:
            for tdomain in cfg.domain.time_domains:
                # Style augmentation parameters
                self.style_params["{}-{}-mean".format(wdomain, tdomain)] = nn.Parameter(torch.zeros(cfg.domain.aug_dim))
                self.style_params["{}-{}-std".format(wdomain, tdomain)] = nn.Parameter(torch.ones(cfg.domain.aug_dim))
                # Addition augmentation parameters
                self.semantic_aug["{}-{}".format(wdomain, tdomain)] = nn.Parameter(torch.zeros(cfg.domain.aug_dim))
        for n, p in self.style_params.named_parameters():
            p.requires_grad = False
        for n, p in self.semantic_aug.named_parameters():
            p.requires_grad = False
    
    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]]=None, use_aug=False):
        losses = {}
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )
        # get outputs from the end of layer 3
        features = self.backbone(images.tensors)
        # apply aug
        # if use_aug:
        #     a_mean, a_std = 
        #     features = features + 
        # compute z: original feature
        features = self.backbone.avgpool(self.backbone.layer4(features))
        """
        # compute z*: z + offset
        loss_domain = 0.0
        for domain, offset in self.offsets.items():
            with torch.no_grad():
                z_star = z + torch.nn.functional.interpolate(offset.unsqueeze(0), z.shape[1], mode="linear").permute(0, 2, 1)[..., None]
                z_star = z_star / z_star.norm(dim=1, keepdim=True)
            # compute z_bar: features that augmented with learnable parameters
            learnable_mean, learnable_std = self.style_params[domain + "-mean"], self.style_params[domain + "-std"]
            learnable_aug = self.semantic_aug[domain]
            feat_aug = features * learnable_std[None, :, None, None] + learnable_mean[None, :, None, None]
            feat_aug = feat_aug + learnable_aug[None, :, None, None]
            z_bar = self.backbone.avgpool(self.backbone.layer4(feat_aug))
            z_bar = z_bar / z_bar.norm(dim=1, keepdim=True)
            cos_dist = 1 - z_star.squeeze(-1).permute(0, 2, 1).bmm(z_bar.squeeze(-1))
            cos_loss = cos_dist.mean()
            l1_loss = torch.nn.functional.l1_loss(z_bar, z)
            loss_domain += (cos_loss + l1_loss)
        losses["loss_domain"] = loss_domain
        """
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses


if __name__ == "__main__":
    x = torch.randn(2, 3, 300, 224).cuda()
    from config import get_cfg
    cfg = get_cfg()
    model = FasterRCNN_noFPN(cfg).to(torch.device("cuda")).eval()
    out = model(x)
    print(out)
