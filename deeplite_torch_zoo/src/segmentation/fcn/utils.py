import numpy as np

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

    - overall accuracy
    - mean accuracy
    - mean IU
    - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

    - overall accuracy
    - mean accuracy
    - mean IU
    - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def get_log_dir(model_name, config_id, cfg):
    # load config
    # import datetime
    # import pytz
    import os
    import os.path as osp

    import yaml

    name = "MODEL-%s" % (model_name)
    # now = datetime.datetime.now(pytz.timezone('America/Bogota'))
    # name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # create out
    log_dir = osp.join("logs", name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_config():
    return {
        # same configuration as original work
        # https://github.com/shelhamer/fcn.berkeleyvision.org
        1: dict(
            max_iteration=100000,
            lr=1.0e-10,
            momentum=0.99,
            weight_decay=0.0005,
            interval_validate=4000,
        )
    }


def get_cuda(cuda, _id):
    import torch

    if not cuda:
        return torch.device("cpu")
    else:
        return torch.device("cuda:{}".format(_id))


def imshow_label(label_show, alpha=1.0):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (0.0, 0.0, 0.0, 1.0)
    cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.arange(0, len(CLASSES))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(label_show, cmap=cmap, norm=norm, alpha=alpha)
    cbar = plt.colorbar(ticks=bounds)
    cbar.ax.set_yticklabels(CLASSES)


def fileimg2model(img_file, transform):
    import PIL

    img = PIL.Image.open(img_file).convert("RGB")
    img = np.array(img, dtype=np.uint8)
    return transform(img, img)[0]


def run_fromfile(model, img_file, cuda, transform, val=False):
    import matplotlib.pyplot as plt
    import torch

    if not val:
        img_torch = torch.unsqueeze(fileimg2model(img_file, transform), 0)
    else:
        img_torch = img_file
    img_torch = img_torch.to(cuda)
    model.eval()
    with torch.no_grad():
        if not val:
            img_org = plt.imread(img_file)
        else:
            img_org = transform(img_file[0], img_file[0])[0]

        score = model(img_torch)
        lbl_pred = score.data.max(1)[1].cpu().numpy()

        plt.imshow(img_org, alpha=0.9)
        imshow_label(lbl_pred[0], alpha=0.5)
        plt.show()
