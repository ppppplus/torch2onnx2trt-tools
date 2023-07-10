
from __future__ import print_function

import numpy as np
import tensorrt as trt
from PIL import ImageDraw
import sys, os
import cv2
import torch
import common
from time import time
# sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 1, 800, 1280]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
def nms_fast(in_corners, H, W, dist_thresh):
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

params = {
        "weights_path": '/home/plus/Work/plvins_ws/src/PL-VINS/feature_tracker/scripts/superpoint/superpoint_v1.pth',
        "max_length": 5,  # Maximum length of point tracks
        "nms_dist": 4,    # Non Maximum Suppression (NMS) distance
        "conf_thresh": 0.015, # Detector confidence threshold
        "nn_thresh": 0.7, # Descriptor matching threshold 
        "cuda": False, # Use cuda GPU to speed up network processing speed
        "H": 800, # Input image height
        "W": 1280, # Input image width
        "min_cnt": 150,
        "trt": True,
        "engine_file_path": "/home/plus/tensorrt/tensorrt/trt_models/superpoint.trt"
}
H = params["H"]
W = params["W"]
conf_thresh = params["conf_thresh"]
nms_dist = params["nms_dist"]
cell = 8
border_remove = 4
onnx_file_path = "/home/plus/tensorrt/tensorrt/trt_models/superpoint.onnx"
engine_file_path = "/home/plus/tensorrt/tensorrt/trt_models/superpoint.trt"
img_path = "/home/plus/tensorrt/tensorrt/img/test.jpg"
img = cv2.imread(img_path )[:, :, ::-1]
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# input = torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.
input = gray_img.astype('float32')[None, None]/255.
print(gray_img.shape)
# Output shapes expected by the post-processor
# output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
# Do inference with TensorRT
trt_outputs = []
with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Do inference
    print("Running inference on image {}...".format(img_path))
    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    inputs[0].host = input
    start_time = time()
    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    end_time = time()
    # print(trt_outputs[0].shape)
    # print(trt_outputs[1].shape)
    # line_segments = trt_outputs[0].reshape((2,2,...))
    # line_desc = trt.outputs[1].reshape((128,...))
    # print(line_segments.shape, line_desc.shape)
    semi = trt_outputs[0].reshape((1,65,100,160))
    coarse_desc = trt_outputs[1].reshape((1,256,100,160))
    print(semi.shape, coarse_desc.shape)
    print("inference time:", end_time-start_time)
# print(type(semi))
semi = semi.squeeze()
dense = np.exp(semi) # Softmax.
dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
# Remove dustbin.
nodust = dense[:-1, :, :]
# Reshape to get full resolution heatmap.
Hc = int(H / cell)
Wc = int(W / cell)
nodust = nodust.transpose(1, 2, 0)
heatmap = np.reshape(nodust, [Hc, Wc, cell, cell])
heatmap = np.transpose(heatmap, [0, 2, 1, 3])
heatmap = np.reshape(heatmap, [Hc*cell, Wc*cell])
xs, ys = np.where(heatmap >= conf_thresh) # Confidence threshold.
pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
pts[0, :] = ys
pts[1, :] = xs
pts[2, :] = heatmap[xs, ys]
pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist) # Apply NMS.
inds = np.argsort(pts[2,:])
pts = pts[:,inds[::-1]] # Sort by confidence.
# Remove points along border.
bord = border_remove
toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
toremove = np.logical_or(toremoveW, toremoveH)
pts = pts[:, ~toremove]
# --- Process descriptor.
D = coarse_desc.shape[1]
if pts.shape[1] == 0:
    desc = np.zeros((D, 0))
else:
    # Interpolate into descriptor map using 2D point locations.
    samp_pts = torch.from_numpy(pts[:2, :].copy())
    samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
    samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
    samp_pts = samp_pts.transpose(0, 1).contiguous()
    samp_pts = samp_pts.view(1, 1, -1, 2)
    samp_pts = samp_pts.float()
    samp_pts = samp_pts.cuda()
    desc = torch.nn.functional.grid_sample(torch.from_numpy(coarse_desc).cuda(), samp_pts)
    desc = desc.data.cpu().numpy().reshape(D, -1)
    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
print(pts.shape, desc.shape)