 
################################################################################
# Modify from : 
# https://github.com/Gwencong/yolov7-pose-tensorrt
# https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
# https://github.com/WongKinYiu/yolov7
# https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose
################################################################################

import cv2
import sys
import random
import platform
import matplotlib
import numpy as np
import configparser

import gi
from gi.repository import Gst
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

categories = ['person']

## gst
 

def make_element(factoryname, name, detail=""):
    """ Creates an element with Gst Element Factory make.
        Return the element  if successfully created, otherwise print
        to stderr and return None.
    """
    print("Creating {}({}) \n".format(name, factoryname))
    elm = Gst.ElementFactory.make(factoryname, name)
    if not elm:
        sys.stderr.write("Unable to create {}({}) \n".format(name, factoryname))
        if detail:
            sys.stderr.write(detail)
    return elm

def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

def create_source_bin(index, uri, file_loop:bool=False):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    if file_loop:
        # use nvurisrcbin to enable file-loop
        uri_decode_bin=Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("file-loop", 1)
        uri_decode_bin.set_property("cudadec-memtype", 0)
    else:
        uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target(
            "src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def set_tracker_config(cfg_path, tracker):
    config = configparser.ConfigParser()
    config.read(cfg_path)
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)
        if key == 'tracking-id-reset-mode' :
            tracking_id_reset_mode = config.getint('tracker', key)
            tracker.set_property('tracking-id-reset-mode', tracking_id_reset_mode)
        if key == 'display-tracking-id' :
            display_tracking_id = config.getint('tracker', key)
            tracker.set_property('display-tracking-id', display_tracking_id)

## postprocess

def get_outshape(input_shape,stride=8,num_calsses=57,anchor_num=3):
    b,c,h,w = input_shape
    s = stride
    height_out = int(h/s)
    width_out = int(w/s)
    out_shape = (b,anchor_num,width_out,height_out,num_calsses)
    return out_shape

def get_total_outshape(input_shape,stride=(8,16,32,64),num_calsses=57,anchor_num=3):
    b,c,h,w = input_shape
    height_out = [int(h/s) for s in stride]
    width_out = [int(w/s) for s in stride]
    num_grids = sum([i*j for i,j in zip(height_out,width_out)])
    num_preds = anchor_num*num_grids
    out_shape = (b,num_preds,num_calsses)
    return out_shape

def postprocess(model_out,im0_shape,im1_shape,im0=None,conf_thres=0.25,iou_thres=0.65,draw=True,line_thickness=3):
    pred = non_max_suppression(model_out, conf_thres, iou_thres)
    # print(f'pred after nms {len(pred)}') 
    boxes,confs,kpts = [], [], []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(im1_shape, det[:, :4], im0_shape, kpt_label=False)
            scale_coords(im1_shape, det[:, 6:], im0_shape, kpt_label=True, step=3)
            
            boxes.append(det[:,:4])
            confs.append(det[:,4:5])
            kpts.append(det[:,6:].reshape(det.shape[0],-1,3))
            # Write results
            if not draw or im0 is None: 
                continue
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                c = int(cls)  # integer class
                label = f'{categories[c]}:{conf:.2f}'
                kpt = det[det_index, 6:]
                plot_one_box(xyxy, im0, colors(c, True), label, line_thickness, kpt_label=True, kpts=kpt, steps=3, orig_shape=im0.shape[:2])
    return boxes, confs, kpts

def postprocessNoNMS(model_out,im0_shape,im1_shape,im0=None,conf_thres=0.5,iou_thres=0.4,draw=True,line_thickness=3):
    # pred = non_max_suppression(model_out, conf_thres, iou_thres)
    nmsed_indices,nmsed_boxes,nmsed_poses,nmsed_scores = model_out
    batch_index, class_index, box_index = nmsed_indices[...,0],nmsed_indices[...,1],nmsed_indices[...,2]
    if np.any(np.isnan(box_index)) or np.all(box_index < 0):
        return [], [], []
    keep_num = np.unique(box_index).size
    # print(box_index)
    assert (box_index[...,keep_num:]-box_index[...,keep_num-1]).sum()==0,f'{box_index}'
    nmsed_boxes  = nmsed_boxes[:,:keep_num,:]
    nmsed_poses  = nmsed_poses[:,:keep_num,:]
    nmsed_scores = nmsed_scores[:,:keep_num,:]
    boxes,confs,kpts = [], [], []
    batch_size = nmsed_boxes.shape[0]
    for batch_id in range(batch_size):  # detections per image
        bboxes = nmsed_boxes[batch_id]
        bboxes = xywh2xyxy(bboxes)
        pposes = nmsed_poses[batch_id]
        scores = nmsed_scores[batch_id]
        num_person = len(bboxes)
        if num_person>0:
            # Rescale boxes from img_size to im0 size
            scale_coords(im1_shape, bboxes, im0_shape, kpt_label=False)
            scale_coords(im1_shape, pposes, im0_shape, kpt_label=True, step=3)
            
            boxes.append(bboxes)
            confs.append(scores)
            kpts.append(pposes.reshape(num_person,-1,3))
            # Write results
            if not draw or im0 is None: 
                continue
            for det_index in range(num_person):
                xyxy = bboxes[det_index]
                conf = scores[det_index]
                kpt = pposes[det_index]
                c = 0  # yolo pose just has one class `person`
                label = f'{categories[c]}:{conf.item():.2f}'
                plot_one_box(xyxy, im0, colors(c, True), label, line_thickness, kpt_label=True, kpts=kpt, steps=3, orig_shape=im0.shape[:2])
    return boxes, confs, kpts


def non_max_suppression(predictions, conf_thres=0.5, nms_thres=0.4):
    """
    description: Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    param:
        prediction: detections, (x1, y1, x2, y2, conf, cls_id)
        origin_h: original image height
        origin_w: original image width
        conf_thres: a confidence threshold to filter detections
        nms_thres: a iou threshold to filter detections
    return:
        boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
    """
    output = [np.zeros((0, 57))] * predictions.shape[0]
    for i,prediction in enumerate(predictions):
        # Get the boxes that score > CONF_THRESH
        # print(f' max : {np.max(prediction[:, 4])}, min : {np.min(prediction[:, 4])}, median: {np.median(prediction[:, 4])}')
        boxes = prediction[prediction[:, 4] >= conf_thres]

        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        # print(f'boxes before xywh2xyxy | max_x :{np.max(boxes [:, 0])}, min_x : {np.min((boxes [:, 0]))}')
        # print(f'boxes before xywh2xyxy | max_y :{np.max(boxes [:, 1])}, min_y : {np.min((boxes [:, 1]))}')
        # print(f'boxes before xywh2xyxy | max_w :{np.max(boxes [:, 2])}, min_w : {np.min((boxes [:, 2]))}')
        # print(f'boxes before xywh2xyxy | max_h :{np.max(boxes [:, 3])}, min_h : {np.min((boxes [:, 3]))}')
        # print(f'kps_x before xywh2xyxy | mean_x  :{np.mean(boxes [:, 6::3])}, max_x  :{np.max(boxes [:, 6::3])}, min_x : {np.min((boxes [:, 6::3]))}')
        # print(f'kps_y before xywh2xyxy | mean_y  :{np.mean(boxes [:, 7::3])}, max_y  :{np.max(boxes [:, 7::3])}, min_y : {np.min((boxes [:, 7::3]))}')
        # print(f'kps_con before xywh2xyxy | mean_con  :{np.mean(boxes [:, 8::3])},max_con  :{np.max(boxes [:, 8::3])}, min_con : {np.min((boxes [:, 8::3]))}')
        # print(f'kps_x before xywh2xyxy |  :{boxes [:, 6::3]}')
        # print(f'kps_y before xywh2xyxy |  :{boxes [:, 7::3]}')
        # print(f'kps_con before xywh2xyxy |  :{boxes [:, 8::3]}')
        boxes[:, :4] = xywh2xyxy(boxes[:, :4])

        # Object confidence
        confs = boxes[:, 4] * boxes[:, 5]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            # label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            # invalid = large_overlap & label_match
            invalid = large_overlap
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        # print(f'boxes after np.stack {boxes.shape}')
        output[i] = boxes
    return output

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    description: compute the IoU of two bounding boxes
    param:
        box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
        x1y1x2y2: select the coordinate format
    return:
        iou: computed iou
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                    np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, kpt_label=False, step=2):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    if isinstance(gain, (list, tuple)):
        gain = gain[0]
    if not kpt_label:
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, [0, 2]] /= gain
        coords[:, [1, 3]] /= gain
        clip_coords(coords[: ,0:4], img0_shape)
        #coords[:, 0:4] = coords[:, 0:4].round()
    else:
        coords[:, 0::step] -= pad[0]  # x padding
        coords[:, 1::step] -= pad[1]  # y padding
        coords[:, 0::step] /= gain
        coords[:, 1::step] /= gain
        clip_coords(coords, img0_shape, step=step)
        #coords = coords.round()
    return coords

def clip_coords(boxes, img_shape, step=2):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0::step] = np.clip(boxes[:, 0::step], 0, img_shape[1])
    boxes[:, 1::step] = np.clip(boxes[:, 1::step], 0, img_shape[0])

def decode(model_outs,input_shape=None,strides=[8,16,32,64]):
    anchors = {  8:[ 19,27,   44,40,   38,94   ],
                16:[ 96,68,   86,152,  180,137 ],
                32:[ 140,301, 303,264, 238,542 ],
                64:[ 436,615, 739,380, 925,792 ] }
    decode_outs = []
    for i,out in enumerate(model_outs):
        b,na,nx,ny,nkpt = out.shape
        if input_shape is not None:
            s_x, s_y = input_shape[0]/ny,input_shape[1]/nx
            assert s_x == s_y, f'stride not equal.'
            stride = int(s_x)
        else:
            stride = strides[i]
        kpt_grid_x,kpt_grid_y = np.meshgrid(np.arange(nx),np.arange(ny))
        kpt_grid_x = kpt_grid_x[np.newaxis,np.newaxis,:,:,np.newaxis]
        kpt_grid_y = kpt_grid_y[np.newaxis,np.newaxis,:,:,np.newaxis]
        grid = np.concatenate([kpt_grid_x,kpt_grid_y],axis=-1)
        anchor_grid = np.array(anchors[stride]).reshape(na,-1)
        anchor_grid = anchor_grid[np.newaxis,:,np.newaxis,np.newaxis,:]
        det = out[..., :6]
        kpt = out[..., 6:]
        det = 1/(1 + np.exp(-det))  # sigmoid
        
        det[..., 0:2] = (det[..., 0:2] * 2. - 0.5 + grid) * stride  # xy
        det[..., 2:4] = (det[..., 2:4] * 2) ** 2 * anchor_grid.reshape(1, na, 1, 1, 2) # wh
        
        kpt[..., 0::3] = (kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(17,axis=-1)) * stride  # xy
        kpt[..., 1::3] = (kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(17,axis=-1)) * stride  # xy
        kpt[..., 2::3] = 1/(1 + np.exp(-kpt[..., 2::3]))
        
        y = np.concatenate((det, kpt), axis=-1)
        decode_outs.append(y.reshape(b,-1,nkpt))
    decode_outs = np.concatenate(decode_outs,axis=1)
    return decode_outs


## plot

def plot_one_box(x, im, color=None, label=None, line_thickness=3, kpt_label=False, kpts=None, steps=2, orig_shape=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, (255,0,0), thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)
    if kpt_label:
        plot_skeleton_kpts(im, kpts, steps, orig_shape=orig_shape)

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
        
 
 
def map_to_zero_one(scalar):
    scalar_min = np.min(scalar)
    scalar_max = np.max(scalar)
    mapped = (scalar - scalar_min) / (scalar_max - scalar_min)
    return mapped