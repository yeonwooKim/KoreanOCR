import json
import numpy as np
import tarfile
import gzip
import io
import random
import cairocffi as cairo
from scipy.ndimage import imread
from matplotlib import font_manager
from matplotlib.image import imsave
from scipy.misc import imresize
from PIL import Image

supported_fonts = ["NanumMyeongjo", "NanumGothic", "Gungsuh", "Batang", "Dotum", "SM SSMyungJo Std", "Gulim",
         "NanumGothicCoding"]
supported_weights = ["NORMAL", "BOLD"]

def get_unavailable(fonts):
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    ret = []
    for font in fonts:
        if font not in available_fonts:
            ret.append(font)
    return ret

WIDTH, HEIGHT = 96, 96
surface = cairo.ImageSurface (cairo.FORMAT_RGB24, WIDTH, HEIGHT)
ctx = cairo.Context (surface)
ctx.set_font_size(45)

mat = dict()

def get_text_dim(text):
    extent = ctx.text_extents(text)
    xbearing, ybearing, width, height, xadvance, yadvance = extent
    return width, height

def rgb2gray(rgb):
    if len(rgb.shape) is 3:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    else:
        print ("Not RGB!")
        return rgb

# Generate 64 X 64 image
def generate_mat(target, font, weight="NORMAL"):
    if font not in supported_fonts:
        print("Unsupported font %s" % font)
        exit(-1)
    if weight not in supported_weights:
        print("Unsupported weight %s" % weight)
        exit(-1)
    
    if weight == "BOLD":
        weight = cairo.FONT_WEIGHT_BOLD
    else:
        weight = cairo.FONT_WEIGHT_NORMAL
    
    w, h = get_text_dim(target)
    if h < 40:
        h = 40
    
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.set_source_rgb(0, 0, 0)
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL,
            weight)
    
    x = 48 - (w/2)
    y = 85 - (h/2)
    
    if target == 'j' or target == 'g' or target == 'p' or target == 'q' or target == 'y':
        y -= 4
        
    if target == "《" and font != "SM SSMyungJo Std" and font != "NanumGothic":
        x -= 20
        if font == "NanumGothicCoding":
            x += 10
        
    if target == "》" and font == "NanumGothicCoding":
        x -= 10
        
    if target == "『" and font != "SM SSMyungJo Std" and font != "NanumGothic" and font != "NanumGothicCoding":
        x -= 20
        
    if target == "「" and font != "NanumGothic":
        x -= 20
        if font == "NanumGothicCoding":
            x += 10
    
    ctx.move_to(x, y)
    ctx.show_text(target)

    fb = io.BytesIO()
    surface.write_to_png (fb)
    new_mat = imread(fb)
    fb.close()
    
    return rgb2gray(new_mat)

def get_mat(target, font=None, weight=None):
    global mat
        
    if font == "Gungsuh" and weight == "BOLD":
        weight = "NORMAL"
        
    if font == "SM SSMyungJo Std" or font == "NanumGothicCoding" and target =="·":
        font = "NanumMyeongjo"
    
    if not font in mat :
        mat[font] = dict()
    
    if not weight in mat[font] :
        mat[font][weight] = dict()
        
    if target in mat[font][weight] :
        return mat[font][weight][target]
    
    mat[font][weight][target] = generate_mat(target, font, weight)
    return mat[font][weight][target]

# Slice a target character from 96 X 96
# with sizing and etc.
def slice_img(mat):
    # offset range are set with subtle reason
    
    scale_factor = random.randrange(52, 70)
    x_offset = random.uniform(-4, 4)
    x_start = round(48 - scale_factor / 2 + x_offset)
    x_offset = random.uniform(-4, 4)
    x_end = round(48 + scale_factor / 2 + x_offset)
    y_offset = random.uniform(-3, 3)
    y_start = round(48 - scale_factor / 2 + y_offset)
    y_offset = random.uniform(-3, 3)
    y_end = round(48 + scale_factor / 2 + y_offset)
    sliced = mat[y_start:y_end, x_start:x_end]
    return imresize(sliced, [32, 32])

def save_chset_random(fonts, weights, chsets, chws, path, num):
    assert len(fonts) > 0
    assert len(weights) > 0
    print ("saving into %s..." % path)
    index_data = []

    tar = tarfile.open(path, "w|gz")
    
    chw_sum = 0
    chws = list(chws)
    for i in range(len(chws)):
        chw_sum += chws[i]
        chws[i] = chw_sum
    
    for i in range(num):
        while True:
            font = random.choice(fonts)
            weight = random.choice(weights)
            if font != "Gungsuh" or weight != "BOLD":
                break
                
        ch_ran = random.uniform(0, chw_sum)
        for i, chset in enumerate(chsets):
            if (ch_ran < chws[i]):
                ch = random.choice(chset)
                break
            
        if len(index_data) % 10000 == 0:
            print ("saving %7dth data...\r" % (len(index_data)+1))
        pathname = "%07d.png" % len(index_data)
        sliced = slice_img(get_mat(ch, font, weight))
        ft = io.BytesIO()
        Image.fromarray(sliced, mode='L').save(ft, format="PNG", optimize=True, compress_level=9)
        index_data.append({'path': pathname, 'font': font, 'weight': weight, 'target': ch})
        ti = tarfile.TarInfo(pathname)
        ti.size = ft.getbuffer().nbytes
        if ti.size < 1 :
            print("Error: size too small")
            print(ft)
            ft.close()
        ft.seek(0)
        tar.addfile(ti, ft)
        ft.close()
    
    ti = tarfile.TarInfo("index.json")
    ft = io.BytesIO()
    ft_str = io.TextIOWrapper(ft)
    json.dump(index_data, ft_str, indent=4, sort_keys=True, separators=(',', ':'))
    ft_str.flush()
    ft.seek(0,2)
    ti.size = ft.tell()
    ft.seek(0)
    tar.addfile(ti, ft)
    ft.close()

    tar.close()
    print ("done")