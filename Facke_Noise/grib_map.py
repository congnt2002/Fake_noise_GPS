import numpy as np
from numpy.random import randint as rd
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import cv2 as cv
XBOUND = (-10.0, 10.0, 0.15625)
YBOUND = (-10.0, 10.0, 0.15625)
GRID_H = int((YBOUND[1] - YBOUND[0]) / YBOUND[2])
GRID_W = int((XBOUND[1] - XBOUND[0]) / XBOUND[2])


gp = [[-9.422196680796333, 0.47498455783352256], 
       [-8.711333354585804, 0.47341656778007746], 
       [-7.7113357874332, 0.4712107456289232], 
       [-6.711338220280595, 0.4690049230121076], 
       [-5.711340653127991, 0.46679910086095333], 
       [-5.000476686633192, 0.4652310572564602], 
       [-4.000479119014926, 0.4630254553630948], 
       [-3.000481551280245, 0.4608198534697294], 
       [-2.0004839836619794, 0.45861425157636404], 
       [-1.2896201278781518, 0.4570463648997247], 
       [0.0, 0.0], 
       [-0.28962255909573287, 0.45484128408133984], 
       [0.7103750096866861, 0.45263620279729366], 
       [1.7103725785855204, 0.4504311219789088], 
       [2.421236024587415, 0.44886360643431544], 
       [3.421233593253419, 0.4466584497131407], 
       [4.4212311619194224, 0.4444532934576273], 
       [5.421228730469011, 0.4422481367364526], 
       [6.1320917753037065, 0.4406805685721338], 
       [7.1320893444353715, 0.4384756349027157], 
       [8.132086913567036, 0.4362707012332976]]
def process_grid_map(global_path):

        def get_global_path(gp):
            gp = np.array(gp)
            # find index of values (0,0) in global_path
            index = np.where((gp == [0.0, 0.0]).all(axis=1))
            # get two previous index, current index, 7 next index
            index = index[0][0]
            if index < 2:
                index = 2
            elif index > len(gp) - 7:
                index = len(gp) - 7

            gp = gp[index - 2:index + 8]
            return gp

        def discretize(gp):
            xx, yy = gp[:, 0], gp[:, 1]
            yi = ((yy - YBOUND[0]) / YBOUND[2])
            yi = np.round(yi)
            yi = np.clip(yi, a_min=0, a_max=GRID_H - 1)
            xi = ((xx - XBOUND[0]) / XBOUND[2])
            xi = np.round(xi)
            xi = np.clip(xi, a_min=0, a_max=GRID_W - 1)

            # find yi, xi where xx, yy is (0,0)
            index = np.where((xx == 0) & (yy == 0))[0]
            return (yi, xi), int(index)

        proc_global_path = get_global_path(global_path)
        grid_map_info, center_idx = discretize(proc_global_path)

        center_y = grid_map_info[0][center_idx]
        center_x = grid_map_info[1][center_idx]
        center_y, center_x = int(center_y), int(center_x)

        # angle_to_path processing
        previous_center_y_1, previous_center_x_1 = grid_map_info[0][center_idx - 1], grid_map_info[1][
            center_idx - 1]
        previous_center_y_1, previous_center_x_1 = int(previous_center_y_1), int(previous_center_x_1)

        previous_center_y_2, previous_center_x_2 = grid_map_info[0][center_idx - 2], grid_map_info[1][
            center_idx - 2]
        previous_center_y_2, previous_center_x_2 = int(previous_center_y_2), int(previous_center_x_2)

        grid_map = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        for i in range(len(grid_map_info[0])):
            y_i, x_i = grid_map_info[0][i], grid_map_info[1][i]
            y_i, x_i = int(y_i), int(x_i)
            if y_i == center_y and x_i == center_x:
                grid_map[y_i][x_i] = 255
            elif (y_i == previous_center_y_1 and x_i == previous_center_x_1) or (
                    y_i == previous_center_y_2 and x_i == previous_center_x_2):
                grid_map[y_i][x_i] = 64
            else:
                grid_map[y_i][x_i] = 128

        # convert to numpy array
        grid_map = np.array(grid_map) / 255.0
        grid_map = np.reshape(grid_map, (1, 1, *grid_map.shape)).astype(np.float32)
        return grid_map
def random_noise(grid_map,min,max):
    grid= grid_map.squeeze()
    center = np.where(grid == 1.0)
    previos_center = np.where(grid == 64/255)
    point = np.where(grid == 128/255)
    coor_y = np.hstack((center[0],previos_center[0],point[0]))
    coor_x = np.hstack((center[1],previos_center[1],point[1]))
    for i in range(len(coor_y)):
        noise = rd(min,max+1)
        coor_x[i] = coor_x[i] + noise
        coor_y[i] = coor_y[i] + noise
    coor = np.column_stack((coor_y, coor_x))
    return coor
def origin(grid_map):
    grid= grid_map.squeeze()
    center = np.where(grid == 1.0)
    previos_center = np.where(grid == 64/255)
    point = np.where(grid == 128/255)
    coor_y = np.hstack((center[0],previos_center[0],point[0]))
    coor_x = np.hstack((center[1],previos_center[1],point[1]))
    coor = np.column_stack((coor_y, coor_x))
    return coor
def vizualize(coor):
    image_size = (128, 128)
    image = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(image)
    points = [tuple(i) for i in coor]
    line_color = (255, 0, 0)  
    point_color = (0, 255, 0) 
    for point in points:
        print(point)
        draw.point(point, fill=point_color)
    draw.line(points, fill=line_color, width=1)
    image.show()

grid = process_grid_map(gp)
#coor = random_noise(grid,-3,3)
coor = origin(grid)
vizualize(coor)

