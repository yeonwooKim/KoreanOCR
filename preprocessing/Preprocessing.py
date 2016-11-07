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
from PIL import Image, ImageDraw, ExifTags
from scipy.ndimage.filters import rank_filter

'''
name: denoising
function: denoise the given image
input: rawimg(image array)
output: normal_img(denoised image)
'''
def denoising(rawimg):
    # normalize the image
    dst = numpy.zeros(rawimg.shape)
    normal_img = cv2.normalize(rawimg, dst, alpha=20, beta=255, norm_type=cv2.NORM_MINMAX)
    #cv2.imwrite('contrast_sample_0.png', normal_img)

    # denoise image: mean_bilateral'
    #bilat_img = cv2.bilateralFilter(normal_img, 3, 20, s0=10, s1=10)
    #cv2.imwrite('bilateral_sample_0.png', bilat_img)    

    # algorithm below is too slow
    #denoiseimg = cv2.fastNlMeansDenoising(contrastimg,None,10,7,21)
    
    return normal_img
    """
    hsv_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0,0,100])
    upper_bound = np.array([255,255,200])
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    denoised = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    cv2.imwrite('denoised_sample_10.png', denoised)

    return denoised
    """
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
    binary_img = cv2.adaptiveThreshold(rawimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                          cv2.THRESH_BINARY_INV, 11, 2)
    return binary_img

'''
' name: find_boundingrect
' function: calculate bounding rectangles from given contours &
			calculate the number of pixels in eact contour
' input: contours(list of contours in the image),
		 edges(edge-detected image)
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
' input: contours(list of contours in the image),
		 edges(edge-detected image)
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
' input: contours(list of contours in the image),
		 edges(edge-detected image) 
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
        box = cv2.cv.BoxPoints(r)
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
        expanded_image = expand_image(edges, N=3, iterations=n) * 255
        _, contours, _ = cv2.findContours(expanded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * expanded_image).show()
    return contours

#Find rects of text areas and Returns a list of (x1, y1, x2, y2) tuples.
'''
' name: find_optimal_bounding_boxes
' function: find bounding rectangles which covers most pixels & 
			use compact area
' input: contours(list of contours in the image),
		 edges(edge-detected image)
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

'''
' name: open_image
' function: open image from the given path
' input: path of the image (+ solve auto-rotated problem of PIL Library)
' output: opened image
'''
def open_image(path):
    image = Image.open(path)
    #image = cv2.imread(path, 0)
    #image = Image.fromarray(numpy.uint8(image))
    # solve auto-rotated problem after Image.open()
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

    return image

'''
' name: shrink_image
' function: shrink image(denoise & easily processed)
' input: image to shrink
' output: shrinked image
'''
def shrink_image(image):
    MAX_DIM = 2048
    w, h = image.size
    if max(w, h) <= MAX_DIM:
        return 1.0, image

    scale = 1.0 * MAX_DIM / max(w, h)
    new_image = image.resize((int(w * scale), int(h * scale)), Image.ANTIALIAS)
    return scale, new_image

'''
' name: preprocess_image
' function: make & store cropped image from the given raw image
' input: path to load image from, path to store image to
' output: None
'''
def preprocess_image(path, out_path):
    #denoised_img = denoising(arr_img)
    #binary_img = thresholding(denoised_img)
    original_image = open_image(path)
    
    scale, shrink_img = shrink_image(original_image)
    shrink_arr = numpy.asarray(shrink_img)

    #gray_img = cv2.cvtColor(arr_img ,cv2.COLOR_BGR2GRAY, gray_img)
    edges = cv2.Canny(shrink_arr, 100, 200)

    _,contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borders = lookup_borders(contours, edges)
    borders.sort(key=(lambda i, x1, y1, x2, y2: (x2 - x1) * (y2 - y1)))

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
        # make cutting image from original 
        rect = [int(x / scale) for x in rect]
        text_img = original_image.crop(rect)
        # if you want to make cutting image from downscaled gray_image, use these codes
        # text_image = gray_image.crop(rect)
        text_img = text_img.convert('L')
        text_arr = numpy.asarray(text_img)
        #(width, height) = text_img.size
        #text_arr = list(text_img.getdata())
        #text_arr = numpy.array(text_arr, dtype=numpy.uint8)
        #text_arr = text_arr.reshape((height, width, 3))
        denoised_img = denoising(text_arr)
        binary_img = thresholding(denoised_img)
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
        out_path = out_path + '_crop.png'
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
