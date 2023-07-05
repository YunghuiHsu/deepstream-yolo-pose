
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

Prequisites:
- DeepStreamSDK 6.2
- Python 3.8
- Gst-python
- GstRtspServer



Installing GstRtspServer and introspection typelib
===================================================
$ sudo apt update
$ sudo apt install python3-gi python3-dev python3-gst-1.0 -y
$ sudo apt-get install libgstrtspserver-1.0-0 gstreamer1.0-rtsp
For gst-rtsp-server (and other GStreamer stuff) to be accessible in
Python through gi.require_version(), it needs to be built with
gobject-introspection enabled (libgstrtspserver-1.0-0 is already).
Yet, we need to install the introspection typelib package:
$ sudo apt-get install libgirepository1.0-dev
$ sudo apt-get install gobject-introspection gir1.2-gst-rtsp-server-1.0

To get test app usage information:
-----------------------------------
  $ python3 deepstream_test1_rtsp_in_rtsp_out.py -h
  
To run the test app with default settings:
------------------------------------------
  1) NVInfer
      $ python3 deepstream_test1_rtsp_in_rtsp_out.py -i rtsp://sample_1.mp4 rtsp://sample_2.mp4 rtsp://sample_N.mp4  -g nvinfer
  2) NVInferserver
      bash /opt/nvidia/deepstream/deepstream-<Version>/samples/prepare_ds_trtis_model_repo.sh
      $ python3 deepstream_test1_rtsp_in_rtsp_out.py -i rtsp://sample_1.mp4 rtsp://sample_2.mp4 rtsp://sample_N.mp4  -g nvinferserver
  
Default RTSP streaming location:
  rtsp://<server IP>:8554/ds-test

This document shall describe the sample deepstream_test1_rtsp_in_rtsp_out application.

This sample app is derived from the deepstream-test3 and deepStream-test1-rtsp-out

This sample app specifically includes following : 
  - Accepts RTSP stream as input and gives out inference as RTSP stream
  - User can choose NVInfer and NVInferserver as GPU inference engine

  If NVInfer is selected then : 
    For reference, here are the config files used for this sample :
    1. The 4-class detector (referred to as pgie in this sample) uses
        dstest1_pgie_config.txt
    2. This 4 class detector detects "Vehicle , RoadSign, TwoWheeler, Person".


    In this sample, first create one instance of "nvinfer", referred as the pgie.
    This is our 4 class detector and it detects for "Vehicle , RoadSign, TwoWheeler,
    Person".

  If NVInferserver is selected then:
    1. Uses SSD neural network running on Triton Inference Server
    2. Selects custom post-processing in the Triton Inference Server config file
    3. Parses the inference output into bounding boxes
    4. Performs post-processing on the generated boxes with NMS (Non-maximum Suppression)
    5. Adds detected objects into the pipeline metadata for downstream processing
    6. Encodes OSD output and shows visual output over RTSP.

