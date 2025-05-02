import importlib.util
spec = importlib.util.spec_from_file_location("cv2", "/usr/local/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38-aarch64-linux-gnu.so")
cv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv2)
import numpy as np
import threading
import time
import math
import matplotlib.pyplot as plt
import traceback


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

LANE_WIDTH_METERS = 0.3  # Width of the lane in meters
XM_PER_PIX = 0.0004  # Meters per pixel in the x dimension (based on camera position and angle)

def init():
    # GStreamer pipeline for the Raspberry Pi Camera
    GST_PIPELINE = (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, "
        "format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, "
        "format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Cannot open the camera")
        exit()
    
    return cap

def cleanup_and_exit(cap):
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Camera capture thread
def capture_frames(cap):
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 0)

        # Store the latest frame in the shared variable
        with frame_lock:
            latest_frame = frame

        # Add a short sleep to prevent excessive resource usage
        time.sleep(0.001)


# Function to convert image to grayscale
def grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian blur to reduce noise and smoothen the image
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Canny edge detection
def canny(image):
    return cv2.Canny(image, 110, 180)

# Mask region of interest
def region_of_interest(image, og_image):
    height, width = image.shape
    mask = np.zeros_like(image)

    # Define the region of interest
    polygon = np.array([[
        (int(width * 0.0), height),
        (int(width * 1.0), height),
        (int(width * 0.9), int(height * 0.42)),
        (int(width * 0.1), int(height * 0.42))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cv2.polylines(og_image, polygon, isClosed=True, color=(0, 255, 0), thickness=3) # visualize the polygon to the screen
    return cv2.bitwise_and(image, mask)

def separate_lane_lines(lane_points, img_width):
    midpoint = img_width // 2
    left_points = []
    right_points = []

    for x, y in lane_points:
        if x < midpoint:
            left_points.append((x, y))
        else:
            right_points.append((x, y))
    
    return left_points, right_points

def get_lane_pixels(binary_warped):
    # Just a simple pixel extraction using contours
    contours, _ = cv2.findContours(binary_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lane_points = []
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            lane_points.append((x, y))
    return lane_points

def fit_poly(lane_points):
    if len(lane_points) < 2:
        return None

    # Extract x and y
    x = np.array([p[0] for p in lane_points])
    y = np.array([p[1] for p in lane_points])

    # Fit a second degree polynomial y = Ax² + Bx + C
    fit = np.polyfit(y, x, 2)  # Fitting x = f(y)
    return fit

def calculate_curvature(poly_coeffs, y_eval):
    A, B, _ = poly_coeffs
    # Radius of curvature: (1 + (2Ay + B)^2)^1.5 / abs(2A)
    return ((1 + (2*A*y_eval + B)**2)**1.5) / np.absolute(2*A)

def draw_poly_curve(img, poly_coeffs):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    fitx = poly_coeffs[0]*ploty**2 + poly_coeffs[1]*ploty + poly_coeffs[2]
    
    points = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.int32)
    cv2.polylines(img, np.int32([points]), isClosed=False, color=(0,255,0), thickness=5)
    return img

def perspective_transform(image, M=None):

    src_rel = np.float32([
        [0.67, 0.45],  # Top-left  <-- Swapped
        [0.33, 0.45],  # Top-right <-- Swapped
        [0.92, 1.00],  # Bottom-left <-- Swapped
        [0.08, 1.00]   # Bottom-right <-- Swapped
    ])

    dst_rel = np.float32([
    [0.25, 0.],  # Top-left (aligned to new top)
    [0.75, 0.],   # Top-right (aligned to new top)
    [0.15, 1.],  # Bottom-left (aligned to new bottom)
    [0.85, 1.]  # Bottom-right (aligned to new bottom)
    ])

    if M is None:
        src = np.float32([
        [x * IMAGE_WIDTH, y * IMAGE_HEIGHT] for x, y in src_rel
        ])

        dst = np.float32([
        [x * IMAGE_WIDTH, y * IMAGE_HEIGHT] for x, y in dst_rel
        ])

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

    if image is not None:
        warped_img = cv2.warpPerspective(image, M, (IMAGE_WIDTH, IMAGE_HEIGHT))
    else:
        warped_img = None

    return warped_img, M

def draw_lane_points(image, lane_points, color=(0, 0, 255), radius=3):
    """
    Draws lane points as small circles on a copy of the image.

    Parameters:
    - image: input image (BGR)
    - lane_points: list of (x, y) tuples
    - color: BGR color of the points (default: red)
    - radius: radius of the points

    Returns:
    - image with drawn lane points
    """
    if lane_points is None:
        return image
    
    img_copy = image.copy()

    for (x, y) in lane_points:
        cv2.circle(img_copy, (int(x), int(y)), radius, color, -1)
    return img_copy

# Measure lane pixel width in bird's eye view (at bottom of warped image)
def get_pixel_to_meter_ratio(left_fit, right_fit, img_height):
    """
    This function calculates the pixel-to-meter ratio based on the known lane width in meters.
    It uses the polynomial coefficients of the left and right lane lines to find their positions at the bottom of the image.
    This only needs to be done once, than just write this returned value in XM_PER_PIX constant.
    """
    y_eval = img_height - 1
    x_left = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    x_right = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_width_pixels = abs(x_right - x_left)
    xm_per_pix = LANE_WIDTH_METERS / lane_width_pixels
    return xm_per_pix

def get_lateral_deviation(left_fit, right_fit, img_width, img_height, xm_per_pix):
    y_eval = img_height - 1
    x_left = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    x_right = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center_x = (x_left + x_right) / 2
    vehicle_center_x = img_width / 2

    deviation_pixels = vehicle_center_x - lane_center_x  # +ve = right, -ve = left
    deviation_meters = deviation_pixels * xm_per_pix
    return deviation_meters

def get_lane_yaw_angle(fit, y_eval):
    # Get dx/dy at bottom of image
    derivative = 2 * fit[0] * y_eval + fit[1]
    angle_rad = math.atan(derivative)  # Slope angle in radians
    return angle_rad  # Can convert to degrees if needed

def get_relative_yaw_angle(left_fit, right_fit, y_eval):
    angle_left = get_lane_yaw_angle(left_fit, y_eval)
    angle_right = get_lane_yaw_angle(right_fit, y_eval)
    avg_lane_angle = (angle_left + angle_right) / 2
    relative_yaw = avg_lane_angle  # Relative to vertical direction (0 rad assumed straight)
    return relative_yaw

def detect(cap):
    M = None
    while cap.isOpened():
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                continue

        #visualize_warp(frame, warped_frame)

        start_time = time.time()
        # Processing each frame
        grey_img = grey(frame)
        blurred_img = gauss(grey_img)
        edges = canny(blurred_img)
        masked_edges = region_of_interest(edges, frame)
        if M is None:
            warped_edges, M = perspective_transform(masked_edges)
        else:
            warped_edges, M = perspective_transform(masked_edges, M)
        
        lane_points = get_lane_pixels(warped_edges)

        left_points, right_points = separate_lane_lines(lane_points, warped_edges.shape[1])

        warped_frame, M = perspective_transform(frame, M)

        #frame = draw_lane_points(warped_frame, right_points)

        left_fit, right_fit = None, None

        if left_points:
            left_fit = fit_poly(left_points)
            if left_fit is not None:
                frame = draw_poly_curve(warped_frame, left_fit)

        if right_points:
            right_fit = fit_poly(right_points)
            if right_fit is not None:
                frame = draw_poly_curve(frame, right_fit)

           
        if left_fit is not None and right_fit is not None:
            #xm_per_pix = get_pixel_to_meter_ratio(left_fit, right_fit, warped_frame.shape[0])
            deviation_meters = get_lateral_deviation(left_fit, right_fit, warped_frame.shape[1], warped_frame.shape[0], XM_PER_PIX)
            cv2.putText(frame, f"Deviation: {deviation_meters:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            rel_yaw_rad = get_relative_yaw_angle(left_fit, right_fit, warped_frame.shape[0])
            rel_yaw_deg = math.degrees(rel_yaw_rad)
            cv2.putText(frame, f"Yaw: {rel_yaw_deg:.2f} degrees", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        #cv2.putText(frame, f"xm_per_pix: {xm_per_pix:.6f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # if poly_coeffs is not None:
        #     draw_poly_curve(frame, poly_coeffs)
        #     # Calculate curvature
        #     curvature = calculate_curvature(poly_coeffs, 250)
        #     cv2.putText(frame, f"Curvature: {curvature:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        end_time = time.time()
        # Calculate FPS
        total_exec_time = end_time - start_time
        fps = 1 / (end_time - start_time)

        # Display FPS on the output image (optional)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Processed Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cleanup_and_exit(cap)

    


latest_frame = None
frame_lock = threading.Lock()

cap = init()

# Start the capture thread
capture_thread = threading.Thread(target=capture_frames, args=(cap,))
capture_thread.daemon = True
capture_thread.start()

try:
    detect(cap)
except Exception as e:
    print("Error in detect: ", e)
    cleanup_and_exit(cap)
    traceback.print_exc()