from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Dict, Tuple, Set, Iterable
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
import matplotlib.cm as cm
import spacy.tokens
import torch
import torch.nn.functional as F
from PIL import Image
from .evaluate import compute_ioa
from .utils import compute_token_merge_indices, cached_nlp, auto_autocast
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

__all__ = ['GlobalHeatMap', 'RawHeatMapCollection', 'WordHeatMap', 'ParsedHeatMap', 'SyntacticHeatMapPair']


def plot_overlay_heat_map(im, heat_map, word=None, out_file=None, crop=None, color_normalize=True, ax=None):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
    if ax is None:
        plt.clf()
        #plt.rcParams.update({'font.size': 24})
        plt_ = plt
    else:
        plt_ = ax
    save_path = 'test'
    
    with auto_autocast(dtype=torch.float32):
        im = np.array(im)

        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im = im[crop:-crop, crop:-crop]

        if color_normalize:
            plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')
        else:
            # heat_map = heat_map.clamp_(min=0, max=255)
            heat_map = heat_map.clamp_(min=0, max=1)
            heat_map = (heat_map * 255).type(torch.uint8)
            #plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet', vmin=0.0, vmax=255.0)
            plt.axis('off')
        # heat_map = (heat_map * 255).type(torch.uint8)

        # threshold_value = 100
        # cococlass = ['background',
        # 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        # 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        # 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        # 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        # 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        # 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        # 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        # 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        # 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        # 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        # 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        # 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        # 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        # cocolist=list(cococlass)
        # img = cv2.imread(heat_map.squeeze().cpu().numpy(), cv2.IMREAD_GRAYSCALE)
        # index=cocolist.index(word)
        # heat_map[heat_map < threshold_value] = 0   # 将小于阈值的像素值置为0
        # heat_map[heat_map >= threshold_value] = 200  #
        # heat_map_np = heat_map.numpy() # 将PyTorch张量转换为NumPy数组
        # cv2.imshow('Heat Map', heat_map_np) # 显示图像 
        # im = torch.from_numpy(im).float() / 255
        # im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1) #torch.cat()是为了把多个tensor进行拼接而存在的
        plt_.imshow(im)
# 显示修改后的图片
        # plt.figure()
#         im_array = np.array(heat_map)
# # 将数组进行二值化
#         threshold = 128  # 设定阈值
#         im_binary = np.where(im_array > threshold, 200, 0)  # 大于阈值的像素设为200，小于阈值的像素设为0
# # 将二值化后的数组转换为PIL Image对象
#         im_binary = Image.fromarray(im_binary.astype('uint8'))
# # 显示二值化后的图片
        # plt.imshow(heat_map_np)
        # plt.axis('off')
        # plt.savefig(os.path.join(save_path, 'ilename'),bbox_inches='tight', pad_inches=0)


        # im_binary = cv2.cvtColor(heat_map, cv2.COLOR_RGB2BGR)
        # output_filename = word + ".png"
        # cv2.imwrite(os.path.join(save_path, output_filename), heat_map_np)
        # cv2.imshow('Image',im_binary)
        plt.axis('off')
        # if word is not None:
        #     if ax is None:
        #         plt.title(word)
        #     else:
        #         ax.set_title(word)

        if out_file is not None:
            plt.savefig(out_file)


class WordHeatMap:
    def __init__(self, heatmap: torch.Tensor, word: str = None, word_idx: int = None):
        self.word = word
        self.word_idx = word_idx
        self.heatmap = heatmap

    @property
    def value(self):
        return self.heatmap

    def plot_overlay(self, image, out_file=None, color_normalize=True, ax=None, **expand_kwargs):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None
        plot_overlay_heat_map(
            image,
            self.expand_as(image, **expand_kwargs),
            word=self.word,
            out_file=out_file,
            color_normalize=color_normalize,
            ax=ax
        )

    def expand_as(self, image, absolute=False, threshold=None, plot=False, **plot_kwargs):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor
        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')

        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        if threshold:
            im = (im > threshold).float()

        im = im.cpu().detach().squeeze()

        if plot:
            self.plot_overlay(image, **plot_kwargs)

        return im

    def compute_ioa(self, other: 'WordHeatMap'):
        return compute_ioa(self.heatmap, other.heatmap)


@dataclass
class SyntacticHeatMapPair:
    head_heat_map: WordHeatMap
    dep_heat_map: WordHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class ParsedHeatMap:
    word_heat_map: WordHeatMap
    token: spacy.tokens.Token


class GlobalHeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.heat_maps = heat_maps
        self.prompt = prompt
        self.compute_word_heat_map = lru_cache(maxsize=50)(self.compute_word_heat_map)

    def compute_word_heat_map(self, word: str, word_idx: int = None, offset_idx: int = 0) -> WordHeatMap:
        merge_idxs, word_idx = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx, offset_idx)
        return WordHeatMap(self.heat_maps[merge_idxs].mean(0), word, word_idx)

    def parsed_heat_maps(self) -> Iterable[ParsedHeatMap]:
        for token in cached_nlp(self.prompt):
            try:
                heat_map = self.compute_word_heat_map(token.text)
                yield ParsedHeatMap(heat_map, token)
            except ValueError:
                pass

    def dependency_relations(self) -> Iterable[SyntacticHeatMapPair]:
        for token in cached_nlp(self.prompt):
            if token.dep_ != 'ROOT':
                try:
                    dep_heat_map = self.compute_word_heat_map(token.text)
                    head_heat_map = self.compute_word_heat_map(token.head.text)

                    yield SyntacticHeatMapPair(head_heat_map, dep_heat_map, token.head.text, token.text, token.dep_)
                except ValueError:
                    pass


RawHeatMapKey = Tuple[int, int, int]  # factor, layer, head


class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        with auto_autocast(dtype=torch.float32):
            key = (factor, layer_idx, head_idx)
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
