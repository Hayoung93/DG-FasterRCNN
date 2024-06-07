import torch
import argparse
from torch.autograd import Variable
from yacs.config import CfgNode as CN
from yacs.config import CfgNode

torch.manual_seed(1)


def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


_C = CN()

_C.data = CN()
_C.data.num_classes = (8                          , "num_classes")
_C.data.data_root   = ("/data/data/DiverseWeather", "data_root")

_C.io = CN()
_C.io.save_dir    = ("./checkpoint/", "save_dir")
_C.io.exp_name    = ("debug"        , "exp_name")
_C.io.resume      = (""             , "resume")
_C.io.learned_aug = (""             , "learned_aug")

_C.model = CN()
_C.model.backbone   = ("resnet101" , "backbone")
_C.model.feat_layer = ("layer3"    , "feat_layer")
_C.model.rpn = CN()
_C.model.rpn.rpn_pre_nms_top_n_train  = (2000, "rpn_pre_nms_top_n_train")
_C.model.rpn.rpn_pre_nms_top_n_test   = (1000, "rpn_pre_nms_top_n_test")
_C.model.rpn.rpn_post_nms_top_n_train = (2000, "rpn_post_nms_top_n_train")
_C.model.rpn.rpn_post_nms_top_n_test  = (2000, "rpn_post_nms_top_n_test")
_C.model.rpn.rpn_nms_thresh           = (0.7, "rpn_nms_thresh")
_C.model.rpn.rpn_fg_iou_thresh        = (0.7, "rpn_fg_iou_thresh")
_C.model.rpn.rpn_bg_iou_thresh        = (0.3, "rpn_bg_iou_thresh")
_C.model.rpn.rpn_batch_size_per_image = (256, "rpn_batch_size_per_image")
_C.model.rpn.rpn_positive_fraction    = (0.5, "rpn_positive_fraction")
_C.model.rpn.rpn_score_thresh         = (0.0, "rpn_score_thresh")
_C.model.box = CN()
_C.model.box.box_score_thresh         = (0.5, "box_score_thresh")
_C.model.box.box_nms_thresh           = (0.5, "box_nms_thresh")
_C.model.box.box_detections_per_img   = (100, "box_detections_per_img")
_C.model.box.box_fg_iou_thresh        = (0.5, "box_fg_iou_thresh")
_C.model.box.box_bg_iou_thresh        = (0.5, "box_bg_iou_thresh")
_C.model.box.box_batch_size_per_image = (512, "box_batch_size_per_image")
_C.model.box.box_positive_fraction    = (0.25, "box_positive_fraction")

_C.transform = CN()
_C.transform.min_size = (800, "min_size")
_C.transform.max_size = (1333, "max_size")

_C.domain = CN()
_C.domain.aug_dim         = (1024                                           , "aug_dim")
_C.domain.weather_domains = (["foggy", "snowy", "cloudy", "rainy", "stormy"], "weather_domains")
_C.domain.time_domains    = (["night", "day", "evening"]                    , "time_domains")

_C.run = CN()
_C.run.batch_size   = (4    , "batch_size")
_C.run.epochs       = (50   , "epochs")
_C.run.num_workers  = (4    , "num_workers")
_C.run.log_interval = (1    , "log_interval")
_C.run.max_norm     = (-1   , "max_norm")
_C.run.minival      = (False, "minival")
_C.run.seed         = (42   , "seed")
_C.run.parallel     = (""   , "parallel")

# optimization settings
_C.run.optim = CN()
_C.run.optim.optimizer    = ("AdamW"            , "optimizer")
_C.run.optim.momentum     = (0.9                , "momentum")
_C.run.optim.nesterov     = ("False"            , "nesterov")
_C.run.optim.lr           = (1e-4               , "lr")
_C.run.optim.lr_backbone  = (1e-4               , "lr_backbone")
_C.run.optim.weight_decay = (1e-5               , "weight_decay")
_C.run.optim.scheduler    = ("CosineAnnealingLR", "scheduler")
_C.run.optim.alpha        = (0.8                , "alpha")
_C.run.optim.beta         = (0.999              , "beta")
_C.run.optim.t_max        = (50                 , "t_max")

# ------ parameters below this line do not have corresponding argparse input ------
# Network settings


def get_cfg_defaults():
    return _C.clone()


def get_cfg():
    parser = argparse.ArgumentParser(description='See configs/train_config.py for detailed information')

    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--learned_aug", type=str)
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--feat_layer", type=str)
    parser.add_argument("--rpn_pre_nms_top_n_train", type=int)
    parser.add_argument("--rpn_pre_nms_top_n_test", type=int)
    parser.add_argument("--rpn_post_nms_top_n_train", type=int)
    parser.add_argument("--rpn_post_nms_top_n_test", type=int)
    parser.add_argument("--rpn_nms_thresh", type=float)
    parser.add_argument("--rpn_fg_iou_thresh", type=float)
    parser.add_argument("--rpn_bg_iou_thresh", type=float)
    parser.add_argument("--rpn_batch_size_per_image", type=int)
    parser.add_argument("--rpn_positive_fraction", type=float)
    parser.add_argument("--rpn_score_thresh", type=float)
    parser.add_argument("--box_score_thresh", type=float)
    parser.add_argument("--box_nms_thresh", type=float)
    parser.add_argument("--box_detections_per_img", type=int)
    parser.add_argument("--box_fg_iou_thresh", type=float)
    parser.add_argument("--box_bg_iou_thresh", type=float)
    parser.add_argument("--box_batch_size_per_image", type=int)
    parser.add_argument("--box_positive_fraction", type=float)
    parser.add_argument("--min_size", type=int)
    parser.add_argument("--max_size", type=int)
    parser.add_argument("--aug_dim", type=int)
    parser.add_argument("--weather_domains", nargs="+", type=str)
    parser.add_argument("--time_domains", nargs="+", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--max_norm", type=int)
    parser.add_argument("--minival", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parallel", type=str)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--nesterov", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_backbone", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--t_max", type=int)


    args, uargs = parser.parse_known_args()
    print("Unknown args: {}".format(uargs))
    cfg = get_cfg_defaults()
    cfg = merge_args(args, cfg)
    return cfg


def merge_args(args, cfg):
    new_cfg = CfgNode({})
    new_cfg.set_new_allowed(True)
    key_gen = get_cfg_keys(cfg, cfg.keys())
    while True:
        try:
            key = next(key_gen)
            value, args_key = eval("cfg." + key)
            if (args_key in args) and (eval("args." + args_key) is not None):
                value = eval("args." + args_key)
            key_split =  key.split(".")
            t1 = {key_split[-1]: value}
            t2 = {}
            for k in key_split[:-1][::-1]:
                if t1 == {}:
                    t1[k] = t2
                    t2 = {}
                else:
                    t2[k] = t1
                    t1 = {}
            if t1 == {}:
                t2 = CfgNode(t2)
                new_cfg = merge_cfg(t2, new_cfg)
            else:
                t1 = CfgNode(t1)
                new_cfg = merge_cfg(t1, new_cfg)
        except StopIteration:
            break
    return new_cfg


def get_cfg_keys(cn, keys):
    for key in keys:
        cur_node = eval("cn." + key)
        if type(cur_node) == CfgNode:
            yield from get_cfg_keys(cn, list(map(lambda x: key + "." + x, cur_node.keys())))
        else:
            yield key

def merge_cfg(a, b):
    for k, v in a.items():
        if k in b:
            if isinstance(v, CfgNode):
                merge_cfg(v, b[k])
            else:
                b[k] = v
        else:
            b[k] = v
    return b
