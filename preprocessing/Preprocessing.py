'''
'   OCR Preprocessing Module
'   Creative Integrated Design M
'   1. denoise an image
'   2. binarize an image
'   3. crop an image according to the text area
'
'   *To Do
'   1. fix the tilted text area
'   2. fix the curved book image
'   3. upgrade denoise algorithm(?)
'''
import os
import sys

import numpy
import cv2
#from PIL import Image, ImageDraw, ExifTags
from scipy.ndimage.filters import rank_filter

'''
' name: 
' function: 
' method: 
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
' name: 
' function: 
' method:
'''
class Paragraph(object):
    def __init__(self, img):
        """set the paragraph image of this class table"""
        self.image = img
    
    def getImage(self):
        """Return the image of this class table"""
        return self.image

'''
name: denoising
function: denoise the given image
input: rawimg(image array)
output: normal_img(denoised image)
'''
def denoising(rawimg):
    # normalize the image
    grayimg = cv2.cvtColor(rawimg,cv2.COLOR_BGR2GRAY)
    mask = numpy.zeros((grayimg.shape),numpy.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    close = cv2.morphologyEx(grayimg,cv2.MORPH_CLOSE,kernel1)
    div = numpy.float32(grayimg)/(close)
    normal_img = numpy.uint8(cv2.normalize(div,div,20,255,cv2.NORM_MINMAX))
    resultimg = cv2.cvtColor(normal_img,cv2.COLOR_GRAY2BGR)
    #cv2.imwrite('morphology_0.png', normal_img)
    
    # denoise image: mean_bilateral'
    #bilat_img = cv2.bilateralFilter(normal_img, 3, 20, s0=10, s1=10)
    #cv2.imwrite('bilateral_sample_0.png', bilat_img)    

    # algorithm below is too slow
    #denoiseimg = cv2.fastNlMeansDenoising(contrastimg,None,10,7,21)
    
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
name: thresholding
function: make a binary image by thrsholding
algorithm: adaptive gaussian thresholding
input: rawimg(image array - np.uint8)
output: binary image
'''
def thresholding(rawimg):
    grayimg = cv2.cvtColor(rawimg,cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                          cv2.THRESH_BINARY_INV, 11, 2)
    resultimg = cv2.cvtColor(binary_img,cv2.COLOR_GRAY2BGR)
    return resultimg

'''
' name: find_boundingrect
' function: calculate bounding rectangles from given contours & calculate the number of pixels in eact contour
' input: contours(list of contours in the image), edges(edge-detected image)
' output: list of bounding rectangle information(x1,y1,x2,y2,pixel-num)
''' 
def find_boundingrect(contours, edges):
    boxes = []
    for c in contours:
        #rect = cv2.minAreaRect(c)
        #box = cv2.boxPoints(rect)
        #box = numpy.int0(box)
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
            print(i, x, y, x+w-1, y+h-1)
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders

'''
' name: remove_border
' function: remove border from the image
' input: contours(list of contours in the image), edges(edge-detected image) 
' output: border-removed image
''' 
def remove_border(contour, edges):
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
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

# Dilate using an NxN [+] sign shape
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
' function: find contours from the given edge-detected image
' input: edges(edge-detected image)
' output: list of contours
'''
def find_connected_components(edges):
    # Perform increasingly aggressive expansion until there are just a few connected components.
    n = 1
    count = 21
    while count > 16:
        n += 1
        expanded_image = expand_image(edges, N=3, iterations=n)
        _, contours, _ = cv2.findContours(expanded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * expanded_image).show()
    return contours

#Find rects of text areas and Returns a list of (x1, y1, x2, y2) tuples.
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
    #print boxes
    
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
    #print optimal_boxes

    return optimal_boxes

def find_vlines(image, w, h):
    V_THRESH = (int)(w * 0.4)
    vlines = []
    for x in range(w):
        y1, y2 = (None,None)
        black = 0
        for y in range(h):
            if pix[x,y] == (0,0,0):
                black = black + 1
                if not y1: y1 = y
                y2 = y
            else:
                if black > V_THRESH:
                    vlines.append((x,y1,x,y2))
                y1, y2 = (None, None)    
                black = 0
        if black > V_THRESH:
            vlines.append((x,y1,x,y2))
    return vlines

def find_hlines(image, w, h):
    H_THRESH = (int)(h * 0.4)
    hlines = []
    for y in range(h):
        x1, x2 = (None, None)
        black = 0
        horiz = []
        for x in range(w):
            if image[x,y] == (0,0,0):
                black = black + 1
                if not x1: x1 = x
                x2 = x
            else:
                if black > H_THRESH:
                    horiz.append((x1,y, x2,y))
                black = 0
                x1, x2 = (None, None)
        if black > H_THRESH:
            horiz.append((x1,y,x2,y))
        if len(horiz) > 0:
            hlines.append(horiz)
    return hlines

def find_cells(vlines, hlines):
    cols = []
    for i in range(0, len(vlines)):
        for j in range(1, len(vlines[i])):
            if vlines[i][j][0] - vlines[i][j-1][0] > 1:
                cols.append((vlines[i][j-1][0],vlines[i][j-1][1],vlines[i][j][2],vlines[i][j][3]))

    rows = []
    for i in range(1, len(hlines)):
        if hlines[i][1] - hlines[i-1][3] > 1:
            rows.append((hlines[i-1][0],hlines[i-1][1],hlines[i][2],hlines[i][3]))
    
    cells = {}
    for i, row in enumerate(rows):
        cells.setdefault(i, {})
        for j, col in enumerate(cols):
            x1 = col[0]
            y1 = row[1]
            x2 = col[2]
            y2 = row[3]
            cells[i][j] = (x1,y1,x2,y2)
    return cells

def extract_table(image):
    w, h = image.shape[0], image.shape[1]
    hlines = get_hlines(image, w, h)
    vlines = get_vlines(image, w, h)
    cells = get_cells(vlines, hlines)
    
    # put the table information in the class and return it

    return

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
                if area >= 10:
                    for p in cnt:
                        points.append(p[0])
                        
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
    
    """
    find the simple contours(four points) of the rotated image
        and do the perspective transform(to fix a curvature)
    
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
    _,contours,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lefttop = (-1, -1)
    leftbottom = (-1, image.shape[1]+1)
    righttop = (image.shape[0]+1, -1)
    rightbottom = (image.shape[0]+1, image.shape[1]+1)
    coord = []
    for h, cnt in enumerate(contours):
        for p in cnt:
            if (lefttop == -1) || (p[0][0] < lefttop[0]):
                if p[0][1] < lefttop[1]:
                    coord.append(p[0])
    """                
    #pts1 = numpy.array([[rect[0], rect[1]], [rect[0]+rect[2], rect[1]], [rect[0], rect[1]+rect[3]], [rect[0]+rect[2],rect[1]+rect[3]]], numpy.float32)
    #pts2 = numpy.array([[rect[0]*50,rect[1]*50],[(rect[0]+rect[2])*50-1,rect[1]*50],[rect[0]*50,(rect[1]+rect[3])*50-1],[(rect[0]+rect[2])*50-1,(rect[1]+rect[3])*50-1]], numpy.float32)
    #retval = cv2.getPerspectiveTransform(pts1,pts2)
    #warp = cv2.warpPerspective(rot_img,retval,(int(rect[2]),int(rect[3])))
    #cv2.imwrite('warpped_0.png', warp)
    
    #cv2.imwrite('ratated_img.png', image)
    
    return image

'''
' name: open_image
' function: open image from the given path
' input: path of the image (+ solve auto-rotated problem of PIL Library)
' output: opened image
'''
def open_image(path):
    #image = Image.open(path)
    image = cv2.imread(path)
    #image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #image = Image.fromarray(numpy.uint8(image))
    # solve auto-rotated problem after Image.open()
    '''
    for orientation in ExifTags.TAGS.keys() : 
        if ExifTags.TAGS[orientation]=='Orientation' : break 
    if (image._getexif() == None):
        return image;
    exif=dict(image._getexif().items())

    if exif[orientation] == 3 : 
        image = image.rotate(180, expand=True)
    elif exif[orientation] == 6 : 
        image = image.rotate(270, expand=True)
    elif exif[orientation] == 8 : 
        image = image.rotate(90, expand=True)
    '''
    return image

'''
' name: shrink_image
' function: shrink image(denoise & easily processed)
' input: image to shrink
' output: shrinked image
'''
def shrink_image(image):
    MAX_DIM = 2048
    if max(image.shape[0], image.shape[1]) <= MAX_DIM:
        return 1.0, image
    
    scale = 1.0 * MAX_DIM / max(image.shape[0], image.shape[1])
    new_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # new_image = image.resize((int(w * scale), int(h * scale)), Image.ANTIALIAS)
    return scale, new_image

'''
' name: preprocess_image
' function: make & store cropped image from the given raw image
' input: path to load image from, path to store image to
' output: None
'''
def preprocess_image(path, out_path):
    original_image = open_image(path)
    #rot_img = rotate_image(original_image)
    scale, shrink_img = shrink_image(original_image)
    
    edges = cv2.Canny(cv2.cvtColor(shrink_img, cv2.COLOR_BGR2GRAY), 100, 200)
    _,contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #rot_img = rotate_image(shrink_img, contours)
    #edges = cv2.Canny(cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY), 100, 200)
    #_,contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    borders = lookup_borders(contours, edges)
    #print(borders)
    borders.sort(key=(lambda x: (x[2]-x[0]) * (x[3]-x[1])))

    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)

        
    edges = 255 * (edges > 0).astype(numpy.uint8)
    #print edges

    # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = numpy.minimum(numpy.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered

    print ('input file: %s' % (path))
    contours = find_connected_components(edges)
    if len(contours) == 0:
        print ('    -> There is no text in the image.')
        return

    #print contours
    
    # 2016-10-15 mjkim. handle multiple text areas
    boxes = find_optimal_bounding_boxes(contours, edges)
    for i, rect in enumerate(boxes):
        """
        make cutting image from original
            if you want to return the image of original size, use below codes
        """
        rect = [int(x / scale) for x in rect]
        text_img = original_image[rect[1]:rect[3], rect[0]:rect[2]]
        
        # text_img = shrink_img[rect[1]:rect[3], rect[0]:rect[2]]
        #text_img = original_image.crop(rect)
        # if you want to make cutting image from downscaled gray_image, use these codes
        # text_image = gray_image.crop(rect)
        
        #text_img = text_img.convert('L')
        #text_arr = numpy.asarray(text_img)
        #(width, height) = text_img
        
        
        #text_arr = list(text_img.getdata())
        #text_arr = numpy.array(text_arr, dtype=numpy.uint8)
        #text_arr = text_arr.reshape((height, width, 3))
        denoised_img = denoising(text_img)
        rot_img = rotate_image(denoised_img)
        binary_img = thresholding(rot_img)
        outfname = '{}_{}.png'.format(out_path, i)
        cv2.imwrite(outfname, binary_img)
        
        print ('    -> %s' % (outfname))
        

if __name__ == '__main__' :
    '''
    path = 'rec_sample_2.jpg'
    outpath = 'rec_crpped_0.png'
    preprocess_image(path, outpath)
    '''
    if len(sys.argv) != 2 :
        print("usage: python Preprocessing.py <path-to-image>")
    else :
        path = sys.argv[1]
        out_path = path[:-4]
        out_path = out_path + '_crop'
        preprocess_image(path, out_path)

"""
    if len(sys.argv) == 2 and '*' in sys.argv[1]:
        files = glob.glob(sys.argv[1])
        random.shuffle(files)
    else:
        files = sys.argv[1:]

    for path in files:
        out_path = path.replace('.jpg', '.crop.png')
        if os.path.exists(out_path): continue
        try:
            process_image(path, out_path)
        except Exception as e:
            print '%s %s' % (path, e)
"""
