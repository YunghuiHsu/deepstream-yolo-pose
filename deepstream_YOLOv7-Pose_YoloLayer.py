#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import argparse
import os
import sys
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'    # Set cuda sync for debug
# sys.path.append("/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/apps")

import numpy as np
from datetime import datetime
import time
import configparser
import math
import platform
import cv2
import pyds

import ctypes
# from common.bus_call import bus_call
from utils.bus_call import bus_call
# from common.is_aarch_64 import is_aarch64
from utils.is_aarch_64 import is_aarch64
# from common.FPS import PERF_DATA
from utils.FPS import PERF_DATA
from utils.utils import make_element,  create_source_bin, set_tracker_config
from utils.utils import map_to_zero_one, postprocess
from utils.display import dispaly_frame_pose, add_obj_meta



import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

# Setting for YOLO-POSE -----------------------------------------------------------------------------
# n_keypoints = 17
# [Optional] yolov8s-pose.engine。(https://github.com/triple-Mu/YOLOv8-TensorRT/tree/triplemu/pose-infer)
# shape_label_pose = 56 # bbox(4) + confidence(1) + keypoints(3 x 17) = 4 + 1 + 0 + 51 = 56
# max_output = 8400
# OUT_SHAPE = (batch_size, shape_label_pose, max_output)  # (batch, 56, 8400)

# [Optional] YOLOv7-pose with YoloLayer_TRT_v7.0 (https://github.com/nanmi/yolov7-pose/)。
# shape_label_pose = 57 # bbox(4) + confidence(1) + cls(1) + keypoints(3 x 17) = 4 + 1 + 1 + 51 = 57
# max_output = 1000
# dim_outputs = n_maxoutput * shape_label_pose + 1  # 57001
# OUT_SHAPE = (batch_size, dim_outputs , 1, 1)


# CONF_THRES = 0.25 # default : 0.25。detect sensitivity. larger, more.
# IOU_THRES = 0.35  # 0.65。detect bboxes overlay tolerance sensitivity. larger, more.
conf_thres = None
iou_thres = None
# ----------------------------------------------------------------------------------------------------

# Setting for DeepStream -----------------------------------------------------------------------------
MAX_DISPLAY_LEN = 64
MAX_TIME_STAMP_LEN = 32
MUXER_OUTPUT_WIDTH = 960  # stream input
MUXER_OUTPUT_HEIGHT = 480  # stream input
# MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280 # stream output
TILED_OUTPUT_HEIGHT = 720 # stream output
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 1

data_type_map = {pyds.NvDsInferDataType.FLOAT: ctypes.c_float,
                    pyds.NvDsInferDataType.INT8: ctypes.c_int8,
                    pyds.NvDsInferDataType.INT32: ctypes.c_int32}
file_loop = False
perf_data = None
# ----------------------------------------------------------------------------------------------------


def pose_src_pad_buffer_probe(pad, info, u_data):
    t = time.time()

    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        pad_index = frame_meta.pad_index
        l_usr = frame_meta.frame_user_meta_list

        while l_usr is not None:
            try:
                # Casting l_obj.data to pyds.NvDsUserMeta
                user_meta = pyds.NvDsUserMeta.cast(l_usr.data)
            except StopIteration:
                break

            # get tensor output
            if (user_meta.base_meta.meta_type !=
                    pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):  # NVDSINFER_TENSOR_OUTPUT_META
                try:
                    l_usr = l_usr.next
                except StopIteration:
                    break
                continue

            try:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(
                    user_meta.user_meta_data)

                # layers_info = []
                # for i in range(tensor_meta.num_output_layers):
                #     layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                #     # print(i, layer.layerName)
                #     layers_info.append(layer)
                    
                assert tensor_meta.num_output_layers == 1, f'Check number of model output layer : {tensor_meta.num_output_layers}'
                # layer_output_info = layers_info[0]
                layer_output_info = pyds.get_nvds_LayerInfo(tensor_meta, 0)  # as num_output_layers == 1

                network_info = tensor_meta.network_info
                input_shape = (network_info.width, network_info.height)
                
                if frame_number == 0 :
                    print(f'\tmodel input_shape  : {input_shape}')
                # print("Network Input : w=%d, h=%d, c=%d"%(network_info.width, network_info.height, network_info.channels))


                # remove zeros from both ends of the array. 'b' : 'both'
                dims = np.trim_zeros(layer_output_info.inferDims.d, 'b')
                
                if frame_number == 0 :
                    print(f'\tModel output dimension from LayerInfo: {dims}')

                    output_message = f'\tCheck model output shape: {layer_output_info.inferDims.numElements}, '
                    output_message += f'given OUT_SHAPE : {dims}'
                    assert layer_output_info.inferDims.numElements == np.prod(dims), output_message
                    
                
                # load float* buffer to python
                cdata_type = data_type_map[layer_output_info.dataType]
                ptr = ctypes.cast(pyds.get_ptr(layer_output_info.buffer), 
                                  ctypes.POINTER(cdata_type))
                # Determine the size of the array
                out = np.ctypeslib.as_array(ptr, shape=dims)

                if frame_number == 0 :
                    print(f'\tLoad Model Output From LayerInfo. Output Shape : {out.shape}')
                    
                # [Optional] Postprocess for yolov8s-pose prediction tensor for YOLOv7-pose with YoloLayer_TRT_v7.0
                # (https://github.com/nanmi/yolov7-pose/)
                # (57001, 1, 1) > (57000, 1, 1) > (1000, 57)。
                out = out[1:, ...].reshape(-1 , 57)   # or out.squeeze().out[:, 1:].reshape(out.shape[0] , -1 , 57)
                # ----------------------------------------------------------------------------------------------------------------------

                #  Explicitly specify batch dimensions
                if np.ndim(out) < 3:
                    out = out[np.newaxis, :]
                    # print(f'add axis 0 for model output : {out.shape}')

                # [Optional] Postprocess for yolov8s-pose prediction tensor 。
                # (https://github.com/triple-Mu/YOLOv8-TensorRT/tree/triplemu/pose-infer)
                # 　(batch, 56, 8400)　＞(batch, 8400, 56) for yolov8
                # out = out.transpose((0, 2, 1))
                # # make pseudo class prob
                # cls_prob = np.ones((out.shape[0], out.shape[1], 1), dtype=np.uint8)
                # out[..., :4] = map_to_zero_one(out[..., :4])  # scalar prob to [0, 1]
                # # insert pseudo class prob into predictions
                # out = np.concatenate((out[..., :5], cls_prob, out[..., 5:]), axis=-1)
                # out[..., [0, 2]] = out[..., [0, 2]] * network_info.width  # scale to screen width
                # out[..., [1, 3]] = out[..., [1, 3]] * network_info.height  # scale to screen height
                # ----------------------------------------------------------------------------------------------------------------------

                output_shape = (MUXER_OUTPUT_HEIGHT, MUXER_OUTPUT_WIDTH)
                if frame_number == 0 :
                    print(f'\tModel output : {out.shape}, The coordinates of the keypoint are rescaled to (h, w) : {output_shape}')
                pred = postprocess(out, output_shape, input_shape,
                                   conf_thres=conf_thres, iou_thres=iou_thres)
                boxes, confs, kpts = pred
                if len(boxes) > 0 and len(confs) > 0 and len(kpts) > 0:
                    add_obj_meta(frame_meta, batch_meta, boxes[0], confs[0])
                    dispaly_frame_pose(frame_meta, batch_meta,
                                       boxes[0], confs[0], kpts[0])

            except StopIteration:
                break

            try:
                l_usr = l_usr.next
            except StopIteration:
                break


        # update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)

        try:
            # indicate inference is performed on the frame
            frame_meta.bInferDone = True
            l_frame = l_frame.next
        except StopIteration:
            break
        # pyds.nvds_acquire_meta_lock(batch_meta)
        # frame_meta.bInferDone = True
        # pyds.nvds_release_meta_lock(batch_meta)

    return Gst.PadProbeReturn.OK


# tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.
# Callback function for deep-copying an NvDsEventMsgMeta struct

# def osd_sink_pad_buffer_probe(pad, info, u_data):
#         buffer = info.get_buffer()
#         batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))

#         l_frame = batch.frame_meta_list
#         while l_frame:
#             frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
#             l_obj = frame_meta.obj_meta_list
#             while l_obj:
#                 obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
#                 obj_meta.text_params.display_text = "person{}: {:.2f}".format(obj_meta.object_id ,obj_meta.tracker_confidence)
#                 obj_meta.text_params.set_bg_clr = 1 # Set boolean indicating that text has bg color to true.
#                 obj_meta.text_params.text_bg_clr.set(0.2, 0.2, 0.2, 0.3) # set(red, green, blue, alpha);
#                 # track_box = obj_meta.tracker_bbox_info.org_bbox_coords
#                 # print(track_box.left,track_box.top,track_box.height,track_box.width)
#                 # rect_params = obj_meta.rect_params
#                 # rect_params.left = track_box.left
#                 # rect_params.top = track_box.top
#                 # rect_params.width = track_box.width
#                 # rect_params.height = track_box.height
#                 l_obj = l_obj.next
#             l_frame = l_frame.next
#         return Gst.PadProbeReturn.OK


def main(args):
    
    global perf_data
    
    # Check input arguments
    number_sources = len(args)
    
    perf_data = PERF_DATA(number_sources)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name, file_loop=file_loop)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    print("Creating Pgie \n ")
    if gie == "nvinfer":
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    else:
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    # print("Creating nvvidconv1 \n ")
    # nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    # if not nvvidconv1:
    #     sys.stderr.write(" Unable to create nvvidconv1 \n")

    # print("Creating filter1 \n ")
    # caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    # filter = Gst.ElementFactory.make("capsfilter", "filter1")
    # if not filter:
    #     sys.stderr.write(" Unable to get the caps filter \n")
    # filter.set_property("caps", caps)

    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd.set_property('process-mode', OSD_PROCESS_MODE)
    nvosd.set_property('display-text', OSD_DISPLAY_TEXT)
    nvvidconv_postosd = Gst.ElementFactory.make(
        "nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    )

    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property("bitrate", bitrate)
    if is_aarch64():
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", 1)
        # encoder.set_property("bufapi-version", 1)

    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")

    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")

    if file_loop:
        if is_aarch64():
            # Set nvbuf-memory-type=4 for aarch64 for file-loop (nvurisrcbin case)
            streammux.set_property('nvbuf-memory-type', 4)
        else:
            # Set nvbuf-memory-type=2 for x86 for file-loop (nvurisrcbin case)
            streammux.set_property('nvbuf-memory-type', 2)
    
    sink.set_property("host", "224.224.255.255")
    sink.set_property("port", updsink_port_num)
    sink.set_property("async", False)
    sink.set_property("sync", 1)
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", number_sources)  # 1 or number_sources
    streammux.set_property("batched-push-timeout", 4000000)

    if gie == "nvinfer":
        pgie.set_property("config-file-path", config_file_path)
    else:
        pgie.set_property("config-file-path",
                          "dstest1_pgie_inferserver_config.txt")

    tracker = make_element("nvtracker", "tracker")
    set_tracker_config("configs/config_tracker.txt", tracker)

    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)

    print("Adding elements to Pipeline \n")
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos", 0)

    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(tiler)
    tiler.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start streaming
    rtsp_port_num = 8554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print(
        "\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n"
        % rtsp_port_num
    )

    # Add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    # either nvosd.get_static_pad("sink") or pgie.get_static_pad("src") works
    pgiepad = pgie.get_static_pad("src")
    if not pgiepad:
        sys.stderr.write(" Unable to get pgiepad src pad of tracker \n")
    pgiepad.add_probe(Gst.PadProbeType.BUFFER, pose_src_pad_buffer_probe, 0)

    # osdpad = nvosd.get_static_pad("sink")
    # if not osdpad:
    #     sys.stderr.write(" Unable to get osdpad sink pad of tracker \n")
    # osdpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pose_src_pad_buffer_probe, 0)
    # perf callback function to print fps every 5 sec
    GLib.timeout_add(5000, perf_data.perf_print_callback)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)

    # start play back and listed to events
    try:
        loop.run()
    except :
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(
        description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input",
                        help="Path to input H264 elementry stream", nargs="+", default=["a"], required=True)
    parser.add_argument("-g", "--gie", default="nvinfer",
                        help="choose GPU inference engine type nvinfer or nvinferserver , default=nvinfer", choices=['nvinfer', 'nvinferserver'])
    parser.add_argument("-c", "--codec", default="H264",
                        help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264', 'H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                        help="Set the encoding bitrate ", type=int)
    parser.add_argument("-config", "--config-file", default='./configs/dstest1_pgie_YOLOv7-Pose-YOLOLAYER_config.txt',
                        help="Set the config file path", type=str)
    parser.add_argument("--file-loop", action="store_true", default=False, dest='file_loop',
                        help="Loop the input file sources after EOS",)
    parser.add_argument("--conf-thres", default=0.25, 
                        help='object confidence threshold', type=float)
    parser.add_argument("--iou-thres", default=0.45, 
                        help='IOU threshold for NMS', type=float)
    # Check input arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global codec
    global bitrate
    global stream_path
    global gie
    global config_file_path
    global file_loop
    global conf_thres
    global iou_thres

    gie = args.gie
    codec = args.codec
    bitrate = args.bitrate
    stream_path = args.input
    config_file_path = args.config_file
    file_loop = args.file_loop
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres
 
    print(f'Args : {args}' ) 
    return stream_path


if __name__ == '__main__':
    stream_path = parse_args()
    sys.exit(main(stream_path))
