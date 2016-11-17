import json
import numpy as np
import tarfile
import gzip
import io
import random
import cairocffi as cairo
from scipy.ndimage import imread
from matplotlib.image import imsave
from scipy.misc import imresize
from PIL import Image
import subprocess
from data import en_chset

supported_fonts = ["NanumMyeongjo", "NanumGothic", "Gungsuh", "Batang", "Dotum", "SM SSMyungJo Std", "Gulim",
         "NanumGothicCoding", "DejaVu Sans", "DejaVu Sans Mono"]
supported_weights = ["NORMAL", "BOLD"]

unsupported_en = dict()
unsupported_kr = dict()
unsupported_weight = dict()
for font in supported_fonts:
    unsupported_en[font] = {}
    unsupported_kr[font] = False
    unsupported_weight[font] = {}

unsupported_en["NanumGothicCoding"] = {"·"}
unsupported_en["SM SSMyungJo Std"] = {"·"}
unsupported_en["DejaVu Sans"] = {"「", "」", "『", "』", "《", "》", "·"}
unsupported_en["DejaVu Sans Mono"] = {"「", "」", "『", "』", "《", "》", "·"}

unsupported_kr["DejaVu Sans"] = True
unsupported_kr["DejaVu Sans Mono"] = True

unsupported_weight["Gungsuh"] = "BOLD"

def refresh_font_cache():
    subprocess.run("fc-cache -f -v", shell=True)

def get_unavailable(fonts):
    p = subprocess.Popen('fc-list -f "%{family}\n"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    lines = p.stdout.read().splitlines()
    available_fonts = { line.decode("utf-8").split(',')[0] for line in lines }
    ret = []
    for font in fonts:
        if font not in available_fonts:
            ret.append(font)
    return ret

WIDTH, HEIGHT = 96, 96
surface = cairo.ImageSurface (cairo.FORMAT_RGB24, WIDTH, HEIGHT)
ctx = cairo.Context (surface)
ctx.set_font_size(45)

__mat = dict()

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
    global __mat
    
    if font is None:
        font = random.choice(supported_fonts)
    if weight is None:
        weight = random.choice(supported_weights)

    if target in unsupported_en[font]:
        return None
    
    if len(target) == 1 and (target not in en_chset) and unsupported_kr[font]:
        return None
    
    if weight in unsupported_weight[font]:
        return None
    
    if not font in __mat :
        __mat[font] = dict()
    
    if not weight in __mat[font] :
        __mat[font][weight] = dict()
        
    if target in __mat[font][weight] :
        return __mat[font][weight][target]
    
    __mat[font][weight][target] = generate_mat(target, font, weight)
    return __mat[font][weight][target]

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

def add_noise(mat):
    background = 255 * np.floor(np.clip(np.random.randn(*mat.shape) - random.random() - 0.8, 0, 1))
    salt = 255 * np.floor(np.clip(np.random.randn(*mat.shape) - random.random() + 0.2, 0, 1))
    cliped = np.clip(mat - background + salt, 0, 255)
    return cliped

def process_mat(mat):
    return slice_img(add_noise(mat))

def get_processed(target, font=None, weight=None):
    mat = get_mat(target, font, weight)
    if mat is None:
        return None
    return process_mat(mat)

def get_inval_char():
    i = random.randrange(0,5)
    if i < 2 :
        return get_inval_num()
    elif i < 4:
        return get_inval_al()
    else :
        return " "

def get_inval_num():
    return random.choice(en_chset[:10]) + random.choice(en_chset[:10])

def get_inval_al():
    return random.choice(en_chset[10:62]) + random.choice(en_chset[10:62])

def save_chset_random(fonts, weights, chsets, chws, path, num):
    assert len(fonts) > 0
    assert len(weights) > 0
    print ("saving %d into %s..." % (num, path))
    index_data = []

    tar = tarfile.open(path, "w|gz")
    
    chw_sum = 0
    chws = list(chws)
    for i in range(len(chws)):
        chw_sum += chws[i]
        chws[i] = chw_sum
    
    i = 0
    while i < num:
        font = random.choice(fonts)
        weight = random.choice(weights)
                
        ch = None
        ch_ran = random.uniform(0, chw_sum)
        for j, chset in enumerate(chsets):
            if (ch_ran < chws[j]):
                ch = random.choice(chset)
                break

        if ch is None:
            ch = get_inval_char()
            
        if i % 10000 == 0:
            print ("saving %7dth data...\r" % (i+1))
        pathname = "%07d.png" % i
        mat = get_mat(ch, font, weight)
        # The font, weight and character combination not supported
        while mat is None:
            font = random.choice(fonts) 
            mat = get_mat(ch, font, weight)

        processed = process_mat(mat)
        ft = io.BytesIO()
        Image.fromarray(processed, mode='L').save(ft, format="PNG", optimize=True, compress_level=9)
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
        i+=1
    
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