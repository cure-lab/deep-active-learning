import numpy as np
import pandas as pd
import torch

def save_df_as_npy(path, df):
    """
    Save pandas dataframe (multi-index or non multi-index) as an NPY file
    for later retrieval. It gets a list of input dataframe's index levels,
    column levels and underlying array data and saves it as an NPY file.

    Parameters
    ----------
    path : str
        Path for saving the dataframe.
    df : pandas dataframe
        Input dataframe's index, column and underlying array data are gathered
        in a nested list and saved as an NPY file.
        This is capable of handling multi-index dataframes.

    Returns
    -------
    out : None

    """

    if df.index.nlevels>1:
        lvls = [list(i) for i in df.index.levels]
        lbls = [list(i) for i in df.index.labels]
        indx = [lvls, lbls]
    else:
        indx = list(df.index)

    if df.columns.nlevels>1:
        lvls = [list(i) for i in df.columns.levels]
        lbls = [list(i) for i in df.columns.labels]
        cols = [lvls, lbls]
    else:
        cols = list(df.columns)

    data_flat = df.values.ravel()
    df_all = [indx, cols, data_flat]
    np.save(path, df_all)

def load_df_from_npy(path):
    """
    Load pandas dataframe (multi-index or regular one) from NPY file.

    Parameters
    ----------
    path : str
        Path to the NPY file containing the saved pandas dataframe data.

    Returns
    -------
    df : Pandas dataframe
        Pandas dataframe that's retrieved back saved earlier as an NPY file.

    """

    df_all = np.load(path)
    if isinstance(df_all[0][0], list):
        indx = pd.MultiIndex(levels=df_all[0][0], labels=df_all[0][1])
    else:
        indx = df_all[0]

    if isinstance(df_all[1][0], list):
        cols = pd.MultiIndex(levels=df_all[1][0], labels=df_all[1][1])
    else:
        cols = df_all[1]

    df0 = pd.DataFrame(index=indx, columns=cols)
    df0[:] = df_all[2].reshape(df0.shape)
    return df0

def max_columns(df0, cols=''):
    """
    Get dataframe with best configurations

    Parameters
    ----------
    df0 : pandas dataframe
        Input pandas dataframe, which could be a multi-index or a regular one.
    cols : list, optional
        List of strings that would be used as the column IDs for
        output pandas dataframe.

    Returns
    -------
    df : Pandas dataframe
        Pandas dataframe with best configurations for each row of the input
        dataframe for maximum value, where configurations refer to the column
        IDs of the input dataframe.

    """

    df = df0.reindex_axis(sorted(df0.columns), axis=1)
    if df.columns.nlevels==1:
        idx = df.values.argmax(-1)
        max_vals = df.values[range(len(idx)), idx]
        max_df = pd.DataFrame({'':df.columns[idx], 'Out':max_vals})
        max_df.index = df.index
    else:
        input_args = [list(i) for i in df.columns.levels]
        input_arg_lens = [len(i) for i in input_args]

        shp = [len(list(i)) for i in df.index.levels] + input_arg_lens
        speedups = df.values.reshape(shp)

        idx = speedups.reshape(speedups.shape[:2] + (-1,)).argmax(-1)
        argmax_idx = np.dstack((np.unravel_index(idx, input_arg_lens)))
        best_args = np.array(input_args)[np.arange(argmax_idx.shape[-1]), argmax_idx]

        N = len(input_arg_lens)
        max_df = pd.DataFrame(best_args.reshape(-1,N), index=df.index)
        max_vals = speedups.max(axis=tuple(-np.arange(len(input_arg_lens))-1)).ravel()
        max_df['Out'] = max_vals
    if cols!='':
        max_df.columns = cols
    return max_df

# For Semi-Supervised
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def linear_rampup(current, rampup_length=200):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha ,lr):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

    def set_wd(self,lr):
        self.wd = 0.02 * lr

# For analysis
class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(self, dataset, preprocess, n_iters, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.n_iters = n_iters

    def __getitem__(self, i):
        x = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess)
        else:
            aug_image = [aug(x, self.preprocess) for i in range(self.n_iters)]
            im_tuple = [self.preprocess(x)] + aug_image
            return im_tuple, i

    def __len__(self):
        return len(self.dataset)

def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
    Returns:
        mixed: Augmented and mixed image.
    """
    from . import augmentations
    from PIL import Image
    image = Image.fromarray(image)
    aug_list = augmentations.augmentations
    # if args.all_ops:
    #     aug_list = augmentations.augmentations_all
    mixture_width, mixture_depth = 1, -1
    aug_severity = 3
    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(
                1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return np.array(mixed)


