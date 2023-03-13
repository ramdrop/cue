from PIL import ImageColor
import numpy as np

HEX2RGB = lambda x: np.array([ImageColor.getcolor(x, 'RGB')])/255.0
RGB = lambda x1, x2, x3: np.array([x1, x2, x3])/255.0

s3dis_map = {
    0: RGB(78, 71, 183),             # ceiling
    1: RGB(152, 223, 138),           # floor
    2: RGB(174, 199, 232),           # wall
    3: RGB(196, 156, 148),           # beam
    4: RGB(66, 188, 102),           # column
    5: RGB(197, 176, 213),           # window
    6: RGB(214, 39, 40),             # door
    7: RGB(188, 189, 34),            # chair
    8: RGB(255, 152, 150),           # table
    9: RGB(148, 103, 189),           # bookcase
    10: RGB(140, 86, 75),            # sofa
    11: RGB(31, 119, 180),           # board
    12: RGB(94, 106, 211),          # clutter
}


scannet_map = {
    0:  RGB(0, 0, 0),
    1:  RGB(174, 199, 232), # wall
    2:  RGB(152, 223, 138), # floor
    3:  RGB(31, 119, 180),  # cabinet
    4:  RGB(255, 187, 120), # bed
    5:  RGB(188, 189, 34),  # chair
    6:  RGB(140, 86, 75),   # sofa
    7:  RGB(255, 152, 150), # table
    8:  RGB(214, 39, 40),   # door
    9:  RGB(197, 176, 213), # window
    10: RGB(148, 103, 189), # bookshelf
    11: RGB(196, 156, 148), # picture
    12: RGB(23, 190, 207),  # counter
    # 13:                   # blinds
    14: RGB(247, 182, 210), # desk
    15: RGB(66, 188, 102),  # shelves
    16: RGB(219, 219, 141), # curtain
    17: RGB(140, 57, 197),  # dresser
    18: RGB(202, 185, 52),  # pillow
    19: RGB(51, 176, 203),  # mirror
    20: RGB(200, 54, 131),  # floor mat
    21: RGB(92, 193, 61),   # clothes
    22: RGB(78, 71, 183),   # ceiling
    23: RGB(172, 114, 82),  # books
    24: RGB(255, 127, 14),  # refridgerato
    25: RGB(91, 163, 138),  # television
    26: RGB(153, 98, 156),  # paper
    27: RGB(140, 153, 101), # towel
    28: RGB(158, 218, 229), # shower curta
    29: RGB(100, 125, 154), # box
    30: RGB(178, 127, 135), # whiteboard
    # 31:                   # person
    32: RGB(146, 111, 194), # nightstand
    33: RGB(44, 160, 44),   # toilet
    34: RGB(112, 128, 144), # sink
    35: RGB(96, 207, 209),  # lamp
    36: RGB(227, 119, 194), # bathtub
    37: RGB(213, 92, 176),  # bag
    38: RGB(94, 106, 211),  # otherstructu
    39: RGB(82, 84, 163),   # otherfurnitu
    40: RGB(100, 85, 144),  # otherprop
    255: RGB(255, 239, 130) #
}

def get_color_map(dataset='scannet'):
    if dataset == 'scannet':
        return scannet_map
    elif dataset == 's3dis':
        return s3dis_map
    else:
        raise Exception('undefined color pattern')


valid_labels = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture')
valid_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, 255)
mapped_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 255)
inverse_map = {mapped_ids[i]:valid_ids[i] for i in range(len(valid_ids))}
def restore_label(label):
    label = np.array([inverse_map[x] for x in label], dtype=np.int32)    # (86854,)
    return label
