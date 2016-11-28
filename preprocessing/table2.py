import sys
import cv2
import numpy as np

def sort_line_list(lines):
    # sort lines into horizontal and vertical
    vertical = []
    horizontal = []
    for line in lines:
        if line[0] == line[2]:
            vertical.append(line)
        elif line[1] == line[3]:
            horizontal.append(line)
    vertical.sort()
    horizontal.sort(key=lambda x: x[1])
    return horizontal, vertical


def remove_duplicates(lines):
    # remove duplicate lines (lines within 10 pixels of eachother)
    for x1, y1, x2, y2 in lines:
        for index, (x3, y3, x4, y4) in enumerate(lines):
            if y1 == y2 and y3 == y4:
                diff = abs(y1-y3)
            elif x1 == x2 and x3 == x4:
                diff = abs(x1-x3)
            else:
                diff = 0
            if diff < 10 and diff is not 0:
                del lines[index]
    return lines

def remove_dup_horiz(lines):
    res = []
    for index, (x1, y1, x2, y2) in enumerate(lines):
        if (index == 0):
            res.append(lines[0])
        else :
            x3, y3, x4, y4 = res[len(res)-1]
            if (x1!=x3 or y1!=y3 or x2!=x4 or y2!=y4):
                if (abs(y1-y3) <= 1) and (x1 >= x3) and (x2 <= x4):
                    continue
                res.append(lines[index])
                
    return res
        
def remove_dup_vert(lines):
    res = []
    for index, (x1, y1, x2, y2) in enumerate(lines):
        if (index == 0):
            res.append(lines[0])
        else :
            x3, y3, x4, y4 = res[len(res)-1]
            if (x1!=x3 or y1!=y3 or x2!=x4 or y2!=y4):
                if (abs(x1-x3) <= 1) and (y1 >= y3) and (y2 <= y4):
                    continue
                res.append(lines[index])
                
    return res
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("1 argument needed: <image path>")
        exit(0)
    
    path = sys.argv[1]
    img = cv2.imread(path)
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(~grayimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2);
    
    horiz_img = binary_img.copy()
    vert_img = binary_img.copy()
    
    scale = 15;
    
    # horizontal lines
    horizontalsize = int(horiz_img.shape[0] / scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize, 1))
    horiz_img = cv2.erode(horiz_img, horizontalStructure, iterations = 1)
    horiz_img = cv2.dilate(horiz_img, horizontalStructure, iterations = 1)
    edges = cv2.Canny(horiz_img, 100, 200, apertureSize=3)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    horiz_lines = []
    mask = np.zeros((img.shape), np.uint8)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt);
        y = int(y + h / 2)
        horiz_lines.append([x,y,x+w,y])
        #cv2.rectangle(mask,(x,y),(x+w,y+h),(0,255,0), 1)
    
    horiz_lines = np.array(horiz_lines).tolist()
    horiz_lines.sort(key=lambda x: x[1])
    horiz_lines = remove_dup_horiz(horiz_lines)
    print(horiz_lines)
    
    for x1, y1, x2, y2 in horiz_lines:
        cv2.line(mask, (x1,y1), (x2,y2), (0,255,0), 1)

    # vertical lines
    verticalsize = int(vert_img.shape[1] / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vert_img = cv2.erode(vert_img, verticalStructure, iterations=1)
    vert_img = cv2.dilate(vert_img, verticalStructure, iterations=1)
    edges = cv2.Canny(vert_img, 100, 200, apertureSize=3)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vert_lines = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt);
        x = int(x + w / 2)
        vert_lines.append([x,y,x,y+h])
        
    vert_lines = np.array(vert_lines).tolist()
    vert_lines.sort()
    vert_lines = remove_dup_vert(vert_lines)

    print("----------------")
    print(vert_lines);
    
    for x1, y1, x2, y2 in vert_lines:
        cv2.line(mask, (x1,y1), (x2,y2), (0,0,255), 1)
    
    
    cv2.imwrite("table2_res.png", mask);
    

