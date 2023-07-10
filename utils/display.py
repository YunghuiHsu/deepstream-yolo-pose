################################################################################
# Modify from : 
# https://github.com/Gwencong/yolov7-pose-tensorrt
# https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
################################################################################

import cv2
import sys
import pyds
import random
import platform
import matplotlib
import numpy as np
from math import ceil

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
# from gi.repository import Gst, GstRtspServer, GLib


def dispaly_frame_pose(frame_meta, batch_meta, boxes, confs, kpts, step=3):
    # add_box_display_meta(frame_meta, batch_meta, boxes, confs,draw_boxes=True, draw_confs=False)
    add_pose_display_meta(frame_meta, batch_meta, kpts,
                          draw_kpts=False, draw_limbs=True)


def add_box_display_meta(frame_meta, batch_meta, boxes, confs, draw_boxes=True, draw_confs=True):
    # Please try to acquire another display meta if the number is bigger than 16.
    need_num = ceil(len(boxes)/16)
    display_metas = [pyds.nvds_acquire_display_meta_from_pool(
        batch_meta) for i in range(need_num)]
    display_meta = None
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        if i % 16 == 0:
            display_meta = display_metas[i//16]
            display_meta.num_rects = 1
            display_meta.num_labels = 1
        # boxes
        if draw_boxes:
            rect_params = display_meta.rect_params[i]
            rect_params.left = int(box[0])
            rect_params.top = int(box[1])
            rect_params.width = int(box[2]-box[0])
            rect_params.height = int(box[3]-box[1])
            rect_params.border_width = 2
            rect_params.border_color.set(0, 0, 1.0, 1.0)
            rect_params.has_bg_color = 1
            rect_params.bg_color.set(0.5, 0.5, 0.5, 0.1)
            display_meta.num_rects += 1

        # confs
        if draw_confs:
            conf_text_params = display_meta.text_params[i]
            conf_text_params.display_text = "person:{:.2f}".format(
                conf.squeeze())
            conf_text_params.x_offset = int(box[0])
            conf_text_params.y_offset = max(int(box[1])-20, 0)
            conf_text_params.font_params.font_name = "Serif"
            conf_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            conf_text_params.font_params.font_size = 10
            conf_text_params.set_bg_clr = 1
            conf_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            display_meta.num_labels += 1
    for display_meta in display_metas:
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def add_pose_display_meta(frame_meta, batch_meta, kpts, draw_kpts=True, draw_limbs=True):
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])/255

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7,
                               0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16,
                              16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 2

    for kpt in kpts:
        # keypoints
        if draw_kpts:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_circles = 1
            for j, pt in enumerate(kpt):
                if pt[2] < 0.5:
                    continue
                r, g, b = pose_kpt_color[j]
                cnt = display_meta.num_circles-1
                circle_params = display_meta.circle_params[cnt]
                circle_params.xc = int(pt[0])
                circle_params.yc = int(pt[1])
                circle_params.radius = radius
                circle_params.circle_color.set(r, g, b, 1.0)
                circle_params.has_bg_color = 1
                circle_params.bg_color.set(r, g, b, 1.0)
                display_meta.num_circles += 1
                if display_meta.num_circles >= 16:
                    display_meta = pyds.nvds_acquire_display_meta_from_pool(
                        batch_meta)
                    display_meta.num_circles = 1
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # limbs
        if draw_limbs:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_lines = 1
            for i, (s1, s2) in enumerate(skeleton):
                pt1, pt2 = kpt[s1-1], kpt[s2-1]
                if pt1[2] < 0.5 or pt2[2] < 0.5:
                    continue
                r, g, b = pose_limb_color[i]
                cnt = display_meta.num_lines-1
                line_params = display_meta.line_params[cnt]
                line_params.x1 = int(pt1[0])
                line_params.y1 = int(pt1[1])
                line_params.x2 = int(pt2[0])
                line_params.y2 = int(pt2[1])
                line_params.line_width = 2
                line_params.line_color.set(r, g, b, 1.0)
                display_meta.num_lines += 1

                if display_meta.num_lines >= 16:
                    pyds.nvds_add_display_meta_to_frame(
                        frame_meta, display_meta)
                    display_meta = pyds.nvds_acquire_display_meta_from_pool(
                        batch_meta)
                    display_meta.num_lines = 1
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def add_obj_meta(frame_meta, batch_meta, boxes, confs):
    pyds.nvds_acquire_meta_lock(batch_meta)
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        new_object = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
        new_object.unique_component_id = 1
        new_object.class_id = 0
        new_object.confidence = conf
        new_object.obj_label = 'person'
        new_object.parent = None

        rect_params = new_object.rect_params
        rect_params.left = int(box[0])
        rect_params.top = int(box[1])
        rect_params.width = int(box[2]-box[0])
        rect_params.height = int(box[3]-box[1])
        rect_params.border_width = 3
        rect_params.border_color.set(0, 0, 1, 1.0)
        rect_params.has_bg_color = 1
        rect_params.bg_color.set(0.5, 0.5, 0.5, 0.1)

        text_params = new_object.text_params
        # text_params.display_text = "person{}: {:.2f}".format(new_object.object_id, conf.squeeze())
        text_params.x_offset = int(box[0])
        text_params.y_offset = max(int(box[1])-20, 0)
        # text_params.font_params.font_name = "Serif"
        # text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        # text_params.font_params.font_size = 10
        # text_params.set_bg_clr = 1 # Set boolean indicating that text has bg color to true.
        # text_params.text_bg_clr.set(0.2, 0.2, 0.2, 0.3) # set(red, green, blue, alpha);

        # raw_box = new_object.detector_bbox_info.org_bbox_coords
        # raw_box.left = int(box[0])
        # raw_box.top = int(box[1])
        # raw_box.width = int(box[2]-box[0])
        # raw_box.height = int(box[3]-box[1])
        pyds.nvds_add_obj_meta_to_frame(frame_meta, new_object, None)
    pyds.nvds_release_meta_lock(batch_meta)
