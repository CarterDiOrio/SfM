import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

pipeline = rs.pipeline()
config = rs.config()

# configure realsense
w = 1280
h = 720
fps = 30
config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.color)
intr = profile.as_video_stream_profile().get_intrinsics()

device = cfg.get_device()
advnc_mode = rs.rs400_advanced_mode(device)
print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

depth_sensor = device.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

hole_filter = rs.hole_filling_filter(2)

print("Intrinsics are: ", intr)

align_to = rs.stream.color
align = rs.align(align_to)

frs = []

while True:
    frames = pipeline.wait_for_frames()
    aligned_frame = align.process(frames)
    depth_frame = aligned_frame.get_depth_frame()
    depth_frame = hole_filter.process(depth_frame)
    color_frame = aligned_frame.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_image = cv2.resize(
        depth_image, (w, h), interpolation=cv2.INTER_AREA)
    color_image = np.asanyarray(color_frame.get_data())

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
        depth_image, alpha=0.03), cv2.COLORMAP_JET)

    cv2.imshow("color", color_image)
    cv2.imshow("depth", depth_colormap)

    frs.append((np.copy(color_image), np.copy(depth_image)))

    if cv2.waitKey(1) == ord('q'):
        break

print(f"images in list {len(frs)}")
img = 0
for color, depth in frs:
    cv2.imwrite(f"../recording/color_{img}.png", color)
    cv2.imwrite(f"../recording/depth_{img}.png", depth)
    img += 1
