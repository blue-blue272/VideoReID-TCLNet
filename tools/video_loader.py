from __future__ import print_function, absolute_import

import os
import torch
import functools
import torch.utils.data as data
from PIL import Image


def pil_loader(path, mode):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def video_loader(img_paths, mode, image_loader):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path, mode))
        else:
            return video
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(data.Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths, mode='RGB')

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, pid, camid



class VideoDataset_seg(data.Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        paths, pid, camid = self.dataset[index]
        img_paths, foreground_paths = paths

        if self.temporal_transform is not None:
            self.temporal_transform.randomize_parameters(len(img_paths))
            img_paths = self.temporal_transform(img_paths)
            foreground_paths = self.temporal_transform(foreground_paths)

        clip = self.loader(img_paths, mode='RGB')
        foreground_clip = self.loader(foreground_paths, mode='L')

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            foreground_clip = [self.spatial_transform(img) for img in foreground_clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        foreground_clip = torch.stack(foreground_clip, 0).permute(1, 0, 2, 3)

        return clip, foreground_clip, pid, camid


if __name__ == '__main__':
    import data_manager
    import transforms.spatial_transforms as ST




if __name__ == '__main__':
    import data_manager
    import transforms.spatial_transforms as ST
    import transforms.temporal_transforms as TT
    import torchvision.utils as tu
    from torchnet.logger import VisdomLogger
    import numpy as np
    np.set_printoptions(threshold='nan')

    def show(logger, clips):
        """clips: [T, C, h, w]
        """
        clips = clips.detach().cpu()
        if clips.size(2) != 256:
            clips = F.interpolate(clips, (256, 128), mode='bilinear', align_corners=True)
        elif clips.size(1) == 3:
            mean=torch.tensor([0.485, 0.456, 0.406])
            std=torch.tensor([0.229, 0.224, 0.225])
            clips.mul_(std.view(1, 3, 1, 1)).add_(mean.view(1, 3, 1, 1))

        clips = tu.make_grid(clips, clips.size(0)).numpy()
        clips = np.array(clips * 255, dtype=np.uint8)
        logger.log(clips)
        return

    dataset = data_manager.init_dataset(name='duke_seg')

    spatial_transform_train = ST.Compose([
                ST.Scale((256, 128), interpolation=3),
                ST.RandomHorizontalFlip(),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_train = TT.TemporalRandomCropSeg(size=8, stride=4)

    dataset = VideoDataset_seg(dataset.train_dense, spatial_transform_train, temporal_transform_train)
    clip, pids, camid = dataset[2000]
    print(clip.size()) #[8, T, H, W]
    clip = clip.transpose(0, 1) #[T, 8, H, W]

    vis_img_logger = VisdomLogger('image', port=8000, opts={'title' : 'img'})
    show(vis_img_logger, clip[:,:3])

    vis_img_logger = VisdomLogger('image', port=8000, opts={'title' : 'head'})
    show(vis_img_logger, clip[:,3:4])

    vis_img_logger = VisdomLogger('image', port=8000, opts={'title' : 'upper'})
    show(vis_img_logger, clip[:,4:5])

    vis_img_logger = VisdomLogger('image', port=8000, opts={'title' : 'lower'})
    show(vis_img_logger, clip[:,5:6])

    vis_img_logger = VisdomLogger('image', port=8000, opts={'title' : 'shoes'})
    show(vis_img_logger, clip[:,6:7])
