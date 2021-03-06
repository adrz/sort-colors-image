#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class Img(object):
    def __init__(self, width, height,
                 cols, sort_idx, velocity=.2):
        if len(cols) != len(sort_idx):
            raise ValueError('len(cols) should be equal to len(sort_idx)')
        self.iteration = 0
        self.width = width
        self.height = height
        self.pixels = []
        self.cols = cols

        for i, c in enumerate(sort_idx):
            x, y = (c // self.height, c % self.height)
            dest_x, dest_y = (i // self.height,
                              i % self.height)
            col = self.cols[c, :]
            pixel = Pixel(col, c, x, y, dest_x,
                          dest_y, self.width,
                          self.height,
                          velocity=velocity)
            self.pixels.append(pixel)

    def is_done(self):
        return all([x.is_done() for x in self.pixels])

    def next_frame(self):
        for pixel in self.pixels:
            pixel.next_position()
        self.iteration += 1

    @staticmethod
    def closest_node(node, nodes):
        return nodes[cdist([node], nodes).argmin()]

    def create_frame(self):
        img = np.zeros((self.width, self.height, 3))
        expected_coord = set([(x, y)
                              for x in range(self.width)
                              for y in range(self.height)])
        coordinates = [(x.get_position(), x.ix) for x in self.pixels]
        for i, y in enumerate(coordinates):
            x, ix = y
            print(i)
            xi, yi = x
            col = self.cols[ix, :]
            img[xi, yi] = col
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_HSV2BGR)
        cv2.imwrite('res/img_{}.png'.format(str(self.iteration).zfill(4)),
                    img)


class Pixel(object):
    def __init__(self, bgr, ix, cur_x, cur_y,
                 dest_x, dest_y, width, height,
                 velocity=1):
        self.dest_x = dest_x
        self.dest_y = dest_y
        self.ix = ix
        self.bgr = bgr
        self.height = height
        self.width = width
        self.velocity = velocity
        self.position = np.array([cur_x, cur_y])
        self.destination = np.array([dest_x, dest_y])

    def set_position(self, x, y):
        self.position = np.array([x, y])

    def get_position(self):
        return (self.position[0],
                self.position[1])

    def get_destination(self):
        return (self.destination[0],
                self.destination[1])

    def get_col(self):
        return self.bgr

    def next_position(self):
        if all(self.position == self.destination):
            return
        vec_pos_to_dest = self.destination - self.position
        distance = np.linalg.norm(vec_pos_to_dest)

        if distance < 12:
            self.position == self.destination
            return
        vec_pos_to_dest = vec_pos_to_dest / distance

        # position = self.position + (vec_pos_to_dest *
        #                             max(1.6, self.velocity *
        #                                 distance))
        position = self.position + (vec_pos_to_dest * distance)
        position = np.uint32(np.round(position))
        if position[0] > self.width-1:
            position[0] = self.width-1
        if position[1] > self.height-1:
            position[1] = self.height-1
        if position[0] < 0:
            position[0] = 0
        if position[1] < 0:
            position[1] = 0
        self.position = position
        return

    def is_done(self):
        return all(self.position == self.destination)


hash_colorspace = {'hsv': [cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR],
                   'luv': [cv2.COLOR_BGR2LUV, cv2.COLOR_LUV2BGR],
                   'hls': [cv2.COLOR_BGR2HLS, cv2.COLOR_HLS2BGR],
                   'xyz': [cv2.COLOR_BGR2XYZ, cv2.COLOR_XYZ2BGR],
                   'lab': [cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR],
                   }


def sort_colors(img_path):
    kmeans = KMeans(n_clusters=50)
    bgr_img = cv2.imread(img_path)
    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    width, height, nc = bgr_img.shape
    img = bgr_img.reshape(width*height, nc)

    clusters_ = kmeans.fit_predict(img)
    new_img = np.zeros((width*height, nc))
    for i, x in enumerate(clusters_):
        new_img[i, :] = kmeans.cluster_centers_[x, :]
    img = new_img
    img_sort = np.array(sorted(img, key=lambda k: (k[0], k[2], k[1])))
    idx_sort = sorted(range(len(img)), key=lambda k: (img[k, 0],
                                                      img[k, 2],
                                                      img[k, 1]))
    # idx_sort = list(range(len(img)))
    # idx_sort = list(range(len(img)))
    img_res = img_sort.reshape(width,
                               height,
                               nc)
    new_img = np.zeros((width, height, nc))

    for i, c in zip(idx_sort, img):
        new_img[i//height, i%height, :] = c

    for c, i in enumerate(idx_sort):
        new_img[c // height, c % height, :] = img[i, :]

    img_final = new_img
    img_final = cv2.cvtColor(np.uint8(img_final), cv2.COLOR_HSV2BGR)
    img_res = cv2.cvtColor(np.uint8(img_res), cv2.COLOR_HSV2BGR)
    cv2.imwrite('test_idx_sort.png', img_final)
    cv2.imwrite('test_img_sort.png', img_res)
    image = Img(width, height, img, idx_sort, velocity=.1)
    for i in range(1000):
        image.create_frame()
        image.next_frame()
