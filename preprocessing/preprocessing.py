'''
'   OCR Preprocessing Module
'   Creative Integrated Design M
'
'   <Work Flow>
'   1. denoise an image
'   2. binarize an image
'   3. crop an image according to the text area
'   4. fix rotation of each layout image
'   5. find tables in the image and put them in the table class
'   6. if there are no tables in the image, put the image in the paragraph class
'
'   <Data Structure>
'   1. Table
'      - store table information(cell's lefttop, rightbottom cutting points) & image
'   2. Paragraph
'      - store an image(which does not contain any tables)
'
'''
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy
import cv2
from scipy.ndimage.filters import rank_filter

from util import *

import table

'''
' name: Table
' function: store table's information & image containing table
' member data: info(cell's lefttop, rightbottom cutting points), img
' method: getInfo, getImage
'''
'''
class Table(object):
    def __init__(self, info, img):
        """set the table info in a multi-dimensional list
        set the image list of each cell."""
        self.info = info
        self.image = img

    def getInfo(self):
        """Return the table info of this class"""
        return self.info
    
    def getImage(self):
        """Return the image list of this class table"""
        return self.image
'''

'''
' name: Paragraph
' function: store an image which does not contain any tables
' member data: img
' method: getImage
'''
'''
class Paragraph(object):
    def __init__(self, img):
        """set the paragraph image of this class table"""
        self.image = img
    
    def getImage(self):
        """Return the image of this class table"""
        return self.image
'''
'''
name: denoising
function: denoise the given image
input: rawimg(image array)
output: normal_img(denoised image)
'''
def denoising(rawimg):
    # normalize the image
    grayimg = cv2.cvtColor(rawimg,cv2.COLOR_BGR2GRAY)
    #grayimg = cv2.GaussianBlur(grayimg,(5,5),0)
    mask = numpy.zeros((grayimg.shape),numpy.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    close = cv2.morphologyEx(grayimg,cv2.MORPH_CLOSE,kernel1)
    div = numpy.float32(grayimg)/(close)
    normal_img = numpy.uint8(cv2.normalize(div,div,20,255,cv2.NORM_MINMAX))
    resultimg = cv2.cvtColor(normal_img,cv2.COLOR_GRAY2BGR)
    #cv2.imwrite('morphology_0.png', normal_img)
    
    return resultimg
    
'''
' name: union_rectangles
' function: union two bounding rectangles
' input: r1, r2(rectangles - x1,y1,x2,y2)
' output: unioned rectangle
'''
def union_rectangles(r1, r2):
    return min(r1[0], r2[0]), min(r1[1], r2[1]), max(r1[2], r2[2]), max(r1[3], r2[3])

'''
' name: intersect_rectangles
' function: find an intersection of two bounding rectangles
' input: r1, r2(rectangles - x1,y1,x2,y2)
' output: intersection of rectangles 
''' 
def intersect_rectangles(r1, r2):
    return max(r1[0], r2[0]), max(r1[1], r2[1]), min(r1[2], r2[2]), min(r1[3], r2[3])

'''
' name: check_intersection
' function: check if there is an intersection between two bounding rectangles
' input: r1, r2(rectangles - x1,y1,x2,y2)
' output: boolean
''' 
def check_intersection(r1, r2):
    r = max(r1[0], r2[0]), max(r1[1], r2[1]), min(r1[2], r2[2]), min(r1[3], r2[3])
    return (r[2]>r[0]) and (r[3]>r[1])

'''
name: lsm
function: least square method to calculate linear background
algorithm: least square method
input: grayimg(image array - np.uint8)
output: regressed image
'''
def lsm(grayimg):
    x = numpy.arange(grayimg.shape[1])
    y = numpy.arange(grayimg.shape[0])
    xv, yv = numpy.meshgrid(x, y)
    constant = numpy.ones(xv.shape)
    flatgrid = numpy.stack((xv.flat, yv.flat, constant.flat), axis=1)
    Ap = numpy.linalg.pinv(flatgrid)
    params = numpy.dot(Ap, grayimg.flat)

    param_without_const = numpy.array([params[0], params[1], 0.5])

    lsm_mat = numpy.dot(flatgrid, param_without_const).reshape(grayimg.shape)
    return lsm_mat

'''
name: thresholding
function: make a binary image by thrsholding. If the image is too small, resize it
algorithm: adaptive gaussian thresholding
input: rawimg(image array - np.uint8)
output: binary image
'''
def thresholding(rawimg):
    grayimg = cv2.cvtColor(rawimg,cv2.COLOR_BGR2GRAY)
    #stretched = grayimg
    subtracted = numpy.clip(grayimg - lsm(grayimg), 0, 255).astype(numpy.uint8)
    blur = cv2.GaussianBlur(subtracted,(1,1),0)
    ths, binary_img = cv2.threshold(~blur,32,255,cv2.THRESH_TOZERO)
    binary_img = numpy.clip(binary_img.astype(numpy.float) * 450 / numpy.amax(binary_img), 0, 255).astype(numpy.uint8)
    #binary_img = cv2.adaptiveThreshold(stretched, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 65, -2);
    #binary_img = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                      cv2.THRESH_BINARY_INV, 11, 2)
    #resultimg = cv2.cvtColor(binary_img,cv2.COLOR_GRAY2BGR)
    return binary_img

'''
' name: find_boundingrect
' function: calculate bounding rectangles from given contours & calculate the number of pixels in eact contour
' input: contours(list of contours in the image), edges(edge-detected image)
' output: list of bounding rectangle information(x1,y1,x2,y2,pixel-num)
''' 
def find_boundingrect(contours, edges):
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        image = numpy.zeros(edges.shape)
        cv2.drawContours(image, [c], 0, 255, -1)
        s = numpy.sum(edges * (image > 0))/255
        if s > 5:
            boxes.append({
                'x1': x,
                'y1': y,
                'x2': x + w - 1,
                'y2': y + h - 1,
                'count': s
            })
    return boxes

'''
' name: lookup_borders
' function: find borders(big bounding rectangle) from the image
' input: contours(list of contours in the image), edges(edge-detected image)
' output: list of borders
''' 
def lookup_borders(contours, edges):
    borders = []
    area = edges.shape[0] * edges.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders

'''
' name: remove_border
' function: remove border from the image
' input: contours(list of contours in the image), edges(edge-detected image) 
' output: border-removed image
''' 
def remove_border(contour, edges):
    image = numpy.zeros(edges.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if min(degs % 90, 90 - (degs % 90)) <= 10.0:    # angle from right
        box = cv2.boxPoints(r)
        box = numpy.int0(box)
        cv2.drawContours(image, [box], 0, 255, -1)
        cv2.drawContours(image, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(image, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), 0, 4)

    return numpy.minimum(image, edges)

'''
' name: expand_image
' function: dilate an image using NxN + shape
' input: edges(edge-detected image), N(size of a kernel), iterations(number of iterations)
' output: dilated image
''' 
def expand_image(edges, N, iterations): 
    arr = numpy.zeros((N,N), dtype=numpy.uint8)
    half = int((N-1)/2)
    arr[half,:] = 1
    expanded_image = cv2.dilate(edges / 255, arr, iterations=iterations)

    arr = numpy.zeros((N,N), dtype=numpy.uint8)
    arr[:,half] = 1
    expanded_image = cv2.dilate(expanded_image, arr, iterations=iterations)
    return 255 * (expanded_image > 0).astype(numpy.uint8)

'''
' name: find_connected_components
' function: dilate the image until contours in the image is <= 16
' input: edges(edge-detected image)
' output: list of contours
'''
def find_connected_components(edges):

    expanded_image = expand_image(edges, N=3, iterations=15)
    _, contours, _ = cv2.findContours(expanded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for i in range(5):
    #n = 1
    #count = 65
    #while count > 64:
        #n += 1
        #expanded_image = expand_image(edges, N=3, iterations=n)
        #_, contours, _ = cv2.findContours(expanded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #count = len(contours)
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * expanded_image).show()
    return contours

'''
' name: find_optimal_bounding_boxes
' function: find bounding rectangles which covers most pixels & use compact area
' input: contours(list of contours in the image),edges(edge-detected image)
' output: list of optimal bounding rectangles
'''
def find_optimal_bounding_boxes(contours, edges):
    optimal_boxes = []
    
    boxes = find_boundingrect(contours, edges)
    boxes.sort(key=lambda x: -x['count'])
    total = numpy.sum(edges) / 255
    
    while len(boxes) > 0:
        box = boxes[0]
        del boxes[0]
        rect = box['x1'], box['y1'], box['x2'], box['y2']
        covered_count = box['count']
        
        while covered_count < total:
            bChanged = False
            for j, box2 in enumerate(boxes):
                rect2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
                if check_intersection(rect, rect2):
                    del boxes[j]
                    rect = union_rectangles(rect, rect2)
                    covered_count += box2['count']
                    bChanged = True

            if not bChanged:
                break

        total = total - covered_count
        
        optimal_boxes.append(rect)

    optimal_boxes.sort(key=lambda x: (x[0], x[1]))
    
    return optimal_boxes

'''
' name: find_table_area
' function: find the area that table exists
' input: image
' output: list of images that contain a table
'''
def find_table_area(image):
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_cnt = []
    rects = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        rect = (x, y, x+w, y+h)
        rects.append(rect)
    
    #rects.sort(key = lambda x: (x[1], x[0]))
    rects.sort(key = lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
    
    selected_rect = []
    for rect in rects:
        if (rect[2]-rect[0])*(rect[3]-rect[1]) > image.shape[0]*image.shape[1]*0.1:
            has_intersection = False
            for r in selected_rect:
                if check_intersection(rect, r): 
                    has_intersection = True
                    break
            if has_intersection is False:
                selected_rect.append(rect)
                
    selected_rect.sort(key = lambda x: [x[1], x[0]], reverse = False)
    selected_rect[1:3] = sorted(selected_rect[1:3], key = lambda x: x[0])
    tables = []
    for r in selected_rect:
        #mask = numpy.zeros((image.shape),numpy.uint8)
        #cv2.drawContours(mask, [c], 0, (255,255,255), -1)
        #cv2.drawContours(mask, [c], 0, (0,0,0), 2)
        #table_img = cv2.bitwise_and(image,mask)
        table_rect = (max(0, r[0]-5), max(0, r[1]-5),
                min(image.shape[1], r[2]+5), min(image.shape[0], r[3]+5))
        if table_rect[2] - table_rect[0] < 2 or table_rect[3] - table_rect[1] < 2:
            continue
        tables.append(table_rect)    
    
    return tables

'''
' name: rotate_image
' function: fix a tilted image, if the largest bounding rect is rotated
' input: image(image numpy array), contours(countours found in the image)
' output: rotation-fixed image
'''
def rotate_image(image):
    """
    find the whole contours and make a numpy array
        make a rotated bounding box, and rotate the image
    """
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
    _,contours,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    points = []
    for h, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= 50:
            for p in cnt:
                points.append(p[0])
    
    if len(points) < 1: return image
    points = numpy.array(points)
    rect = cv2.minAreaRect(points)
    """
    DEBUG
    box = cv2.boxPoints(rect)
    box = numpy.int0(box)
    mask = numpy.zeros((image.shape),numpy.uint8)
    cv2.drawContours(mask,[box],0,(0,0,255),2)
    cv2.imwrite('{}_{}.png'.format('rotbox',rect[2]), mask)
    """
    angle = abs(rect[2])
    if angle > 45: angle = angle - 90 
    mat = cv2.getRotationMatrix2D(rect[0], -angle, 1)
    image = cv2.warpAffine(image, mat, (image.shape[1], image.shape[0]), image.size, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, (255,255,255))
    
    return image

'''
' name: open_image
' function: open image from the given path
' input: path of the image (+ solve auto-rotated problem of PIL Library)
' output: opened image
'''
def open_image(path):
    image = cv2.imread(path)
    #image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # solve auto-rotated problem after Image.open()
    return image

'''
' name: shrink_image
' function: shrink image(denoise & easily processed)
' input: image to shrink
' output: shrinked image
'''
def shrink_image(image):
    MAX_DIM = 1024
    if max(image.shape[0], image.shape[1]) <= MAX_DIM:
        return 1.0, image
    
    scale = 1.0 * MAX_DIM / max(image.shape[0], image.shape[1])
    new_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return scale, new_image

'''
' name: stretch_image
' function: stretch image(to improve threshold result)
' input: image to strech
' output: stretched image
'''
def stretch_image(image):
    MIN_DIM = 256
    if min(image.shape[0], image.shape[1]) >= MIN_DIM:
        return 1.0, image
    
    scale = 1.0 * MIN_DIM / min(image.shape[0], image.shape[1])
    new_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return scale, new_image

'''
' name: preprocess_image
' function: make & store cropped image from the given raw image
' input: path to load image from, path to store image to
' output: None
'''
def preprocess_image(img, out_path=None, save=False):
    # If not BGR, make it BGR to avoid error
    if len(img.shape) < 3:
        original_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        original_image = img
    scale, shrink_img = shrink_image(original_image)
    
    edges = cv2.Canny(cv2.cvtColor(shrink_img, cv2.COLOR_BGR2GRAY), 100, 200)
    _,contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    borders = lookup_borders(contours, edges)
    borders.sort(key=(lambda x: (x[2]-x[0]) * (x[3]-x[1])))

    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)
    
        
    edges = 255 * (edges > 0).astype(numpy.uint8)
    
    
    # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = numpy.minimum(numpy.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered
    

    # print ('input file: %s' % (path))
    contours = find_connected_components(edges)
    if len(contours) == 0:
        #print ('    -> There is no text in the image.')
        return

    # 2016-10-15. handle multiple text areas
    boxes = find_optimal_bounding_boxes(contours, edges)
    layouts = []
    for i, rect in enumerate(boxes):
        """
        make cutting image from original
            if you want to return the image of original size, use below codes
        """
        rect = [int(x / scale) for x in rect]
        text_img = original_image[rect[1]:rect[3], rect[0]:rect[2]]
        
        """
        # use the shrinked image as a result
        text_img = shrink_img[rect[1]:rect[3], rect[0]:rect[2]]
        text_img = original_image.crop(rect)
        # if you want to make cutting image from downscaled gray_image, use these codes
        text_image = gray_image.crop(rect)
        """
        
        denoised_img = denoising(text_img)
        #denoised_img = text_img
        rot_img = rotate_image(denoised_img)
        
        tables = find_table_area(rot_img)

        binary_img = thresholding(rot_img)
        
        for j, trect in enumerate(tables):
                t = rot_img[trect[1]:trect[3], trect[0]:trect[2]]
                info = table.find_table(t)
                if save:
                    outfname = '{}_{}_{}_{}.png'.format(out_path, 'table', i, j)
                    cv2.imwrite(outfname, binary_img)
                    print ('    -> %s' % (outfname))
                if info is None:
                    print("table info None")
                    continue

                for row_cells in info:
                    for cell in row_cells:
                        if cell[2] == 1: continue # Merged cell
                        cell_rect = (trect[0]+cell[5], trect[1]+cell[6], trect[0]+cell[7], trect[1]+cell[8])
                        if cell_rect[2] - cell_rect[0] < 2 or cell_rect[3] - cell_rect[1] < 2:
                            continue
                        cell_img = numpy.copy(binary_img[cell_rect[1]:cell_rect[3], cell_rect[0]:cell_rect[2]])
                        parag = Paragraph(img = cell_img, rect = cell_rect)
                        layouts.append(parag)
                binary_img[trect[1]:trect[3], trect[0]:trect[2]].fill(0) # Remove out table area

        parag = Paragraph(img = binary_img, rect = rect)
        layouts.append(parag)
        if save:
            outfname = '{}_{}_{}.png'.format(out_path, 'paragraph', i)
            cv2.imwrite(outfname, binary_img)
            print ('    -> %s' % (outfname))
    #print(layouts)    
    return layouts
        
if __name__ == '__main__' :
    if len(sys.argv) != 2 :
        print("usage: python Preprocessing.py <path-to-image>")
    else :
        path = sys.argv[1]
        out_path = path[:-4]
        out_path = out_path + '_crop'
        layouts = preprocess_image(open_image(path), out_path, save=True)
