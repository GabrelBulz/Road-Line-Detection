import sys
import numpy
# numpy.set_printoptions(threshold=sys.maxsize)
import cv2
from matplotlib import pyplot as plt
import os 
import time
import numpy as np
import math


def create_green(src, dest):
    img  = cv2.imread(src)

    green = np.zeros(img.shape)
    green[:,:,1] = img[:,:,1] 

    cv2.imwrite(dest, green) 



def create_green(img, dest):
    green = np.zeros(img.shape)
    green[:,:,1] = img[:,:,1] 

    cv2.imwrite(dest, green) 



def create_blue(img, dest):
    blue = np.zeros(img.shape)
    blue[:,:,0] = img[:,:,0] 

    cv2.imwrite(dest, blue) 


def create_red(img, dest):

    red = np.zeros(img.shape)
    red[:,:,2] = img[:,:,2] 

    cv2.imwrite(dest, red)
    return red 

def create_canny(src, dest):
    img = cv2.imread(src,0)
    img = cv2.GaussianBlur(img, (7,7), 0)
    # print(img)

    # plt.subplot(121),plt.imshow(img, cmap='gray')
    # plt.show()

    edges = cv2.Canny(img,100,120)
    cv2.imwrite(dest, edges)

def create_treshold(src, dest):
    
    img = cv2.imread(src)
    img = img[:,:,2]
    # print(img)
    # plt.subplot(121),plt.imshow(img)
    # plt.show()
    # img = cv2.medianBlur(img,5)
    # thresh = (62,255)
    # thresh = (55,230)
    thresh = (180,230)
    # shape = img.shape 
    # print(shape)

    output = np.zeros_like(img)
    # for i in range(shape[0]):
    #     for j in range(shape[1]):
    #         if(img[i][j] >= thresh[0] and img[i][j] <= thresh[1]):
    #             output[i][j] = 255
    output[(img >= thresh[0]) & (img <= thresh[1])] = 255


    cv2.imwrite(dest, output)


    # x,y = img.shape
    # cont = 0
    # for i in range(x):
    #     for j in range(y):
    #         if(img[i][j] > 60 and img[i][j] < 100):
    #             img[i][j] = 255
    #         else:
    #             img[i][j] = 0

    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    # plt.show()

def img_to_gray(src, dest):
    img = cv2.imread(src, 0)
    cv2.imwrite(dest, img)

def canny_plus_threshold(canny_src, thresh_src, dest):
    canny = cv2.imread(canny_src, 0)
    thresh = cv2.imread(thresh_src, 0)
    output = np.zeros_like(canny)
    output[(canny == 255) | (thresh == 255)] = 255
    cv2.circle(output, (650,1050), 10, color=(255,255,255))
    cv2.circle(output, (650,1500), 10, color=(255,255,255))
    cv2.imwrite(dest, output)

def crop_img(img, x, y, x1, y1):
    return img[x:x1, y:y1]

def calculate_hist_cols(img):
    hist = []
    shape = img.shape

    for i in range(shape[1]):
        cont = 0
        for j in range(int(shape[0]//3.5), shape[0]):
            cont += img[j][i]/255
        hist.append(cont)

    return hist

def get_xy_unwarped(p, matrix):
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    return int(px),int(py)

def img_to_warp_perspective(img):
    # cv2.circle(img,(800,710),10,color=(255,0,0))
    # cv2.circle(img,(1270,710),10,color=(255,0,0))
    # cv2.circle(img,(250,850),10,color=(255,0,0))
    # cv2.circle(img,(1450,850),10,color=(255,0,0))

    src_pts = np.float32([[800,710],[1270,710],[250,850],[1490,850]])
    dest_pts =  np.float32([[0,0],[1600,0],[0,1200],[1600, 1200]])
    mat_pers = cv2.getPerspectiveTransform(src_pts, dest_pts)
    rev = cv2.getPerspectiveTransform(dest_pts,src_pts)
    # print(mat_pers)
    # print(rev)
    print(get_xy_unwarped(0,0, rev))
    warped = cv2.warpPerspective(img, mat_pers, (1600,1200))

    # cv2.imshow("Ceva", img)
    # # cv2.imshow("2", warped)

    # plt.subplot(121),plt.imshow(img)
    # plt.show()
    # # mat_pers_reverse = cv2.getPerspectiveTransform(dest_pts, src_pts)
    # # reversed = cv2.perspectiveTransform()
    

    return warped


def avg(x):
    s=0
    for el in x:
        s += el[0]

    return s//len(x)
   
def get_lanes_from_histogram(img, pos_start_lane):
    """
        return a list of points from a late
        and modify the image to change the color of lane
    """
    points= []
    shape = img.shape
    y_box_start_values = list(range(0,shape[0],70))
    y_box_start_values.reverse()
    
    
    for y_start in y_box_start_values:
        box_points_list = []
        for x in range(max(0,pos_start_lane - 75), min((shape[1]-1), pos_start_lane + 75)):
            for y in range(min((shape[0]-1),(y_start+70)), y_start, -1):
                if(img[y][x] > 200):
                    box_points_list.append([x,y])
                    img[y][x] = 125
        if(box_points_list != []):
            pos_start_lane = avg(box_points_list)
        points += box_points_list

    return points


def get_xy_from_list(lst):

    result_x = []
    result_y = []
    for el in lst:
            result_x.append(el[0])
            result_y.append(el[1])

    return result_x,result_y

def highlight_lanes(img, img_name):
    shape = img.shape

    # get middle lane
    img_mid = shape[1]//2
    error_section = shape[1]//6
    left_border = img_mid - error_section
    right_border = img_mid + error_section
    hist = calculate_hist_cols(img)
    index_hist_max = hist.index(max(hist[left_border:right_border]))
    middle_lane_points = get_lanes_from_histogram(img, index_hist_max)

    #middle_x, middle_y = get_xy_from_list(middle_lane_points)
    
    np_arr_middle = np.asarray(middle_lane_points)


    #get right lane
    max_right = max(hist[index_hist_max+140:len(hist)])
    index_right = hist.index(max_right)
    
    right_lane_points = get_lanes_from_histogram(img, index_right)

    # if(right_lane_points != []):
    #     right_x, right_y = get_xy_from_list(right_lane_points)
    #     # right_fit = np.polyfit(right_x, right_y, 2)

    np_arr_right = np.asarray(right_lane_points)
    
    color_lanes_original_images(img, img_name, np_arr_middle, np_arr_right)


def color_lanes_original_images(img, img_name, left_line_points, right_line_points):

    # these are the points used for the original warp
    # we will need to appli getPerspective, but with src and dest interchanged
    src_pts = np.float32([[800,710],[1270,710],[250,850],[1490,850]])
    dest_pts =  np.float32([[0,0],[1600,0],[0,1200],[1600, 1200]])

    reverse_mat = cv2.getPerspectiveTransform(dest_pts,src_pts)

    img2 = cv2.imread("..\\day_landscape\\"+img_name, cv2.COLOR_BGR2RGB)

    if(len(left_line_points > 1)):
        left_points = [get_xy_unwarped(points, reverse_mat) for points in left_line_points]
        cv2.polylines(img2, [np.asarray(left_points)], False, color=(0,255,0), thickness=5)

    if(len(right_line_points > 1)):
        right_points = [get_xy_unwarped(points, reverse_mat) for points in right_line_points]
        cv2.polylines(img2, [np.asarray(right_points)], False, color=(0,255,0), thickness=5)

    cv2.imwrite("..\\result_lines\\"+img_name, img2)
    

# img = cv2.imread(".\\day\\threshold.jpeg")
# img = crop_img(img, 1050, 0, 1500, 1080)

# hist = cv2.calcHist([img],[0],None,[256],[0,256])
# hist = [[0],[1],[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]]
# hist = calculate_hist_cols(img)
# print(len(hist))
# plt.plot(hist)
# plt.show()
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.show()
#### green specter showed best results
# create_green(".\\day\\100.jpeg", ".\\day\\green.jpeg")
# create_red(".\\day\\100.jpeg", ".\\day\\red.jpeg")
# create_blue(".\\day\\100.jpeg", ".\\day\\blue.jpeg")


# create_canny(".\\day\\100.jpeg",".\\day\\canny.jpeg")
# img_to_gray(".\\day\\red.jpeg", ".\\day\\gray_red.jpeg")


# create_treshold(".\\day\\gray_red.jpeg", ".\\day\\threshold.jpeg")

# canny_plus_threshold(".\\day\\canny.jpeg", ".\\day\\threshold.jpeg", ".\\day\\res_canny_thresh.jpeg")

# img = cv2.imread(".\\night\\green.jpeg",0)
# edges = cv2.Canny(img,50,100)
# # cv2.imwrite(".\\night\\res_green.jpeg", edges)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()


def process_folder(src):
    os.chdir(src)
    cont = 0
    for img_name in os.listdir():
        img = cv2.imread(img_name)
        number = img_name.split('.')[0]
        red_name = "..\\results_red\\"+str(number)+".jpeg"
        thresh_name = "..\\results_th\\"+str(number)+".jpeg"
        warp_name = "..\\results_warp\\"+str(number)+".jpeg"
        red = create_red(img, red_name)
        warped = img_to_warp_perspective(red)
        cv2.imwrite(warp_name, warped)
        create_treshold(warp_name, thresh_name)
        cont += 1
        print(cont)

# process_folder("C:\\Users\\bulzg\\Desktop\\road_detection\\test\\test")





# split from video to frames
os.chdir("C:\\Users\\bulzg\\Desktop\\road_detection\\day_landscape")
# img = cv2.imread("0.jpeg")
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.show()
# frame = cv2.VideoCapture("day_landscape.mp4")
# succ,img  = frame.read()
# print(succ)
# cont = 0
# while succ:
#     # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#     img = crop_img(img, 0, 0, 880, img.shape[1])


#     cv2.imwrite(".\\day_landscape\\"+ str(cont) + ".jpeg", img) 
#     succ,img = frame.read()
#     cont +=1


# img = cv2.imread("test.jpeg")
# warp = img_to_warp_perspective(img)
# create_canny("warp_color.jpeg", "canny.jpeg")
# create_red(warp, ".red.jpeg")
# create_treshold(".red.jpeg", ".th.jpeg")



# # detect
# cont = 0
# for image_name in os.listdir():
#     img = cv2.imread(image_name, 0)  
#     highlight_lanes(img, image_name)
#     print(image_name)
#     print(cont)
#     cont += 1


cont = 0
for image_name in os.listdir():
    img = cv2.imread(image_name)  
    img2 = crop_img(img, 530, 150, 880, 1600)
    cv2.imwrite("C:\\Users\\bulzg\\Desktop\\road_detection\\day_landscape_crop\\"+ str(image_name.split('.')[0]) + ".jpeg", img2)
    print(cont)
    cont += 1

# print(img.shape)
# print(img.shape)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.show()
# hist = calculate_hist_cols(img)
# plt.plot(hist)
# plt.show()
# plt.plot(hist)
# print(hist.index(max(hist)))
# plt.show()





# index_hist_max = hist.index(max(hist))
# get_lanes_from_histogram(img, index_hist_max)


# plt.plot(hist[0:index_hist_max-100])
# plt.show()
# print(max_left)
# print(max_right)
# print(hist.index(max_left))
# get_lanes_from_histogram(img, hist.index(max(max_left, max_right)))
# red = red