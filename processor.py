#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import os
import math
from tqdm import tqdm

from numba import jit
import warnings

warnings.filterwarnings("ignore")


# In[2]:


@jit(nopython=True)
def d_ar_ll(values, filter_shape, threshold = 10**(-4)):
    if filter_shape == (1, 1):
        return no_compress(values, threshold)
    sp = values.shape
    fs0 = filter_shape[0]
    fs1 = filter_shape[1]
    dim_0_batches = sp[0]/fs0
    dim_1_batches = sp[1]/fs1
    result = [((0.0, 0.0), (0.0, 0.0))]
    for b0 in range(dim_0_batches):
        for b1 in range(dim_1_batches):
            lb0 = b0*fs0
            lb1 = b1*fs1
            res = proc_rect(values[lb0:lb0 + fs0, lb1:lb1 + fs1])
            if(abs(res[0]) > threshold or abs(res[1]) > threshold):
                result.append(((lb0 + fs0/2, lb1 + fs1/2), res))
    return result

@jit(nopython=True)
def no_compress(values, threshold = 10**(-4)):
    sp = values.shape
    result = [((0.0, 0.0), (0.0, 0.0))]
    for b0 in range(sp[0]):
        for b1 in range(sp[1]):
            tmp = values[b0, b1]
#             res.append(tmp[1])
#             res.append(tmp[0])
            res = (tmp[1], tmp[0])
            if(abs(res[0]) > threshold or abs(res[1]) > threshold):
                result.append(((b0, b1), res))
    return result

@jit(nopython=True)
def proc_rect(values):
    res_y = 0
    res_x = 0
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            val = values[y][x]
            res_x += val[1]
            res_y += val[0]
    return (res_x, res_y)


# In[3]:


#function for sign determinition; <0 => -1, =0 => 0, >0 => 1.
@jit(nopython=True)
def sign(x):
    return (x > 0) - (x < 0)

#function for finding vector destination points with their magnitude
@jit(nopython=True)
def n_jit_rad(data, shape):
    result = [(np.array([0.0, 0.0]), 0.0)]
    for item in data[1:]:
        place = item[0]
        velocity = item[1]
        x = velocity[1]
        y = velocity[0]
        magn = round((y**2 + x**2)**(0.5), 4)
        y_sign = sign(y)
        x_sign = sign(x)
        m = shape[1]-1
        n = shape[0]-1
        i = place[1]
        j = place[0]

        if x_sign and y_sign:
            t_x = (-j + m * int(x > 0))/x
            t_y = (-i + n * int(y > 0))/y
            t = min(t_x, t_y)
        elif x_sign:
            t = (-j + m * int(x > 0))/x
        else:
            t = (-i + n * int(y > 0))/y

        #determinition of destiny point

        result_x = round(j + t * x, 2)
        result_y = round(i + t * y, 2)

        key = np.array([result_x, result_y])

        result.append((key, magn))

    return result


# In[4]:


@jit(nopython = True)
def ceil(val):
    return val + (1 - val % 1)

@jit(nopython = True)
def point_range(orient, const, value_below, value_above):
    temp = np.arange(value_below, value_above)
    val_mas = np.full((int(value_above - value_below)), const)
    return np.column_stack((val_mas, temp)) if orient else np.column_stack((temp, val_mas))


@jit(nopython = True)
def neighbours(placement, distance, shape):

    result = []

    placement_0 = placement[0]

    shape_0 = shape[0]
    
    #determining along which dimension neighbours are placed

    move_along = int(placement_0 == 0 or placement_0 == shape_0-1)
    another_direction = abs(move_along - 1)

    placement_of_interest = placement[another_direction]
    shape_of_interest = shape[another_direction]

    placement_of_movement = placement[move_along]
    shape_of_movement = shape[move_along]

    #finding values, in range of which lie all integer points
    value_below = int(math.ceil(placement[move_along] - distance))
    value_above = int(ceil(placement[move_along] + distance))

    #determination whether all the neighbours lie on found edge
    below_ok = value_below >= 0
    above_ok = value_above <= shape[move_along]

    #fulfilling neighbours list
    #if all the neihbours lie on one edge
    if (below_ok and above_ok):
        neighbour = point_range(move_along, placement_of_interest, value_below, value_above)
    
    #if barrier below was met
    elif not below_ok:
        neighbour = point_range(move_along, placement_of_interest, 0, value_above)
        if (placement_of_interest):
            j = float(shape_of_interest - 1)
            corner = np.array([0, 0])
            corner[another_direction] += j
            cathet = np.linalg.norm(corner - placement)
            seeking_cathet = ceil((distance ** 2 - cathet ** 2) ** 0.5)
            additional = point_range(another_direction, 0, shape_of_interest - 1 - seeking_cathet, shape_of_interest - 1)
        else:
            j = 0.0
            corner = np.array([0, 0])
            corner[another_direction] += j
            cathet = np.linalg.norm(np.array([0, 0]) - placement)
            seeking_cathet = ceil((distance ** 2 - cathet ** 2) ** 0.5)
            additional = point_range(another_direction, 0, 1, seeking_cathet)
        neighbour = np.vstack((neighbour, additional))
    #if barrier above was met
    elif not above_ok:
        neighbour = point_range(move_along, placement_of_interest, value_below, shape_of_movement)
        curr_shape = shape_of_movement - 1
        if (placement_of_interest):
            j = float(shape_of_interest - 1)
            corner = np.array([0, 0])
            corner[another_direction] += j
            corner[move_along] += curr_shape
            cathet = np.linalg.norm(corner - placement)
            seeking_cathet = ceil((distance ** 2 - cathet ** 2) ** 0.5)
            additional = point_range(another_direction, curr_shape, shape_of_interest - 1 - seeking_cathet, shape_of_interest - 1)
        else:
            j = 0.0
            corner = np.array([0, 0])
            corner[move_along] += curr_shape
            cathet = np.linalg.norm(corner - placement)
            seeking_cathet = ceil((distance ** 2 - cathet ** 2) ** 0.5)
            additional = point_range(another_direction, curr_shape, 1, seeking_cathet)

        neighbour = np.vstack((neighbour, additional))
    #if both of them were met
    else:
        assert False, "Too big value"

    for n in range(neighbour.shape[0]):
        obj = neighbour[n]
        result.append((obj, round(np.linalg.norm(obj - placement), 2)))

    return result

@jit(nopython = True)
def process(point, distance, val_list, init_shape):
    point_neighbours = neighbours(point[0], 5, init_shape)
    for i in point_neighbours:
        t = int(get_index(i[0], init_shape))
        val_list[t] += point[1] * round(2**(-i[1]), 4)

@jit(nopython = True)
def get_index(dot, shape):
    if dot[0] == 0:
        return dot[1]
    if dot[1] == shape[1] - 1:
        return (shape[1] - 1) + dot[0]
    if dot[0] == shape[0] - 1:
        return (shape[1] - 1) + (shape[0] - 1) + (shape[1] - dot[1] - 1)
    if dot[1] == 0:
        return 2 * (shape[1] - 1) + (shape[0] - 1) + (shape[0] - dot[0] - 1)

@jit(nopython = True)
def pressure(values, distance, init_shape):

    summary_array = np.zeros(2*init_shape[0] + 2*init_shape[1] - 4)
    t = 0
    for i in values[1:]:
        process(i, distance, summary_array, init_shape)
        t+=1
    return summary_array


# In[67]:


# cap = cv.VideoCapture("Building_204_2A.mp4")
# length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# fps = int(cap.get(cv.CAP_PROP_FPS))
# fourcc = cv.VideoWriter_fourcc('m','p','4','v')

# ret, first_frame = cap.read()
# side = 256
# # vw = cv.VideoWriter('trial002.mp4', fourcc, fps, (side, side))

# prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# prev_gray = cv.resize(prev_gray, (side, side)).astype(np.uint8)
# mask = np.zeros((side, side, 3))
# mask[..., 1] = 255

# result = []

# for i in tqdm(range(length-1)):
#     ret, frame = cap.read()

#     frame = cv.resize(frame, (side, side)).astype(np.uint8)
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 3, 3, 9, 1.7, 0)

#     temp = flow[..., 0].copy()
#     flow[..., 0] = flow[..., 1].copy()
#     flow[..., 1] = temp.copy()

# #     magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
# #     mask[..., 0] = angle * 255 / np.pi / 2
# #     mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
      
# #     mask = mask.astype(np.uint8)

# #     rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

#     ll = d_ar_ll(flow, (1, 1))

#     point_press = n_jit_rad(ll, (side, side))

#     result.append(pressure(point_press, 25, (side, side)))

# #     result = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX)

# #     channel_2 = rgb.shape[2]
# #     rgb_0 = rgb.shape[0]
# #     rgb_1 = rgb.shape[1]
# #     rgb[0, : ] = np.repeat(result[:rgb_0], channel_2).reshape(-1, channel_2)
# #     rgb[:, -1, :] = np.repeat(result[rgb_0 - 1:rgb_0 + rgb_1 - 1], channel_2).reshape(-1, channel_2)
# #     rgb[-1, :] = np.repeat(result[rgb_0 + rgb_1 - 2:2 * (rgb_0 - 1) + rgb_1][::-1], channel_2).reshape(-1, channel_2)
# #     rgb[:, 0, :] = np.repeat(np.vstack((result[2 * (rgb_0 - 1) + (rgb_1 - 1):], result[:1]))[::-1], channel_2).reshape(-1, channel_2)

# #     vw.write(rgb)
      
#     prev_gray = gray

# cap.release()
# # vw.release()

# result = np.vstack(tuple(result))
# np.savez_compressed('compressed.npz', result)


# In[6]:


def process_video(path, save_path, side = 256, compressing_shape = (8, 8), neighbour_dist = 25):
    cap = cv.VideoCapture(path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    prev_gray = cv.resize(prev_gray, (side, side)).astype(np.uint8)

    result = []

    for i in range(length-1):
        ret, frame = cap.read()

        frame = cv.resize(frame, (side, side)).astype(np.uint8)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 3, 3, 9, 1.7, 0)

        temp = flow[..., 0].copy()
        flow[..., 0] = flow[..., 1].copy()
        flow[..., 1] = temp.copy()

        ll = d_ar_ll(flow, compressing_shape)

        point_press = n_jit_rad(ll, (side, side))

        result.append(pressure(point_press, neighbour_dist, (side, side)))

        prev_gray = gray

    cap.release()

    result = np.vstack(tuple(result))
    file_name = path.split('/')[-1]
    resulting_path = save_path + file_name[:-4] + '.npz'
    np.savez_compressed(resulting_path, result)


# In[7]:


# process_video("Building_204_2A.mp4")


# In[ ]:




