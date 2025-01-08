import importlib.util
spec = importlib.util.spec_from_file_location("cv2", "/usr/local/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38-aarch64-linux-gnu.so")
cv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv2)
import numpy as np
import threading
import time


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
    return cv2.Canny(image, 120, 170)

# Mask region of interest
def region_of_interest(image, og_image):
    height, width = image.shape
    mask = np.zeros_like(image)

    # Define the region of interest
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.62), int(height * 0.42)),
        (int(width * 0.32), int(height * 0.42))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cv2.polylines(og_image, polygon, isClosed=True, color=(0, 255, 0), thickness=3) # visualize the polygon to the screen
    return cv2.bitwise_and(image, mask)

# Extract lane pixels using non-zero values
def extract_lane_pixels(image):
    non_zero_pixels = np.argwhere(image > 0)
    y_vals = non_zero_pixels[:, 0]
    x_vals = non_zero_pixels[:, 1]
    return x_vals, y_vals

# Fit polynomial to lane pixels
def fit_polynomial(x_vals, y_vals):
    if len(x_vals) > 0 and len(y_vals) > 0:
        return np.polyfit(y_vals, x_vals, 2)
    else:
        return None

# Calculate curvature radius
def calculate_curvature(y_eval, fit_coeffs):
    if fit_coeffs is not None:
        A = fit_coeffs[0]
        B = fit_coeffs[1]
        R_curve = ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / np.abs(2 * A)
        return R_curve
    return None

# Draw lanes on the image
def draw_lane(image, left_fit, right_fit):
    max_y_value = 550
    y_vals = np.linspace(0, image.shape[0] - 1, image.shape[0])


    y_vals = y_vals[y_vals >= max_y_value]
    
    if left_fit is not None:
        left_x_vals = left_fit[0] * y_vals**2 + left_fit[1] * y_vals + left_fit[2]
    else:
        left_x_vals = None
        
    if right_fit is not None:
        right_x_vals = right_fit[0] * y_vals**2 + right_fit[1] * y_vals + right_fit[2]
    else:
        right_x_vals = None

    # Calculate the center curve as the average of left and right curves
    if left_x_vals is not None and right_x_vals is not None:
        center_x_vals = (left_x_vals + right_x_vals) / 2
    else:
        center_x_vals = None  
    
    # Draw the left curve
    if left_x_vals is not None:
        for i in range(1, len(y_vals)):
            cv2.line(image, (int(left_x_vals[i - 1]), int(y_vals[i - 1])), 
                     (int(left_x_vals[i]), int(y_vals[i])), (0, 255, 0), 3)

    # Draw the right curve
    if right_x_vals is not None:
        for i in range(1, len(y_vals)):
            cv2.line(image, (int(right_x_vals[i - 1]), int(y_vals[i - 1])), 
                     (int(right_x_vals[i]), int(y_vals[i])), (0, 255, 0), 3)

    # Draw the center curve
    if center_x_vals is not None:
        for i in range(1, len(y_vals)):
            cv2.line(image, (int(center_x_vals[i - 1]), int(y_vals[i - 1])), 
                     (int(center_x_vals[i]), int(y_vals[i])), (255, 0, 255), 2)  # Drawing the center in magenta

    return image

def detect(cap):
    while cap.isOpened():
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                continue

        start_time = time.time()
        # Processing each frame
        grey_img = grey(frame)
        blurred_img = gauss(grey_img)
        edges = canny(blurred_img)
        masked_edges = region_of_interest(edges, frame)

        # Extract lane pixels
        left_pixels = masked_edges[:, :masked_edges.shape[1]//2]
        right_pixels = masked_edges[:, masked_edges.shape[1]//2:]
        
        left_x_vals, left_y_vals = extract_lane_pixels(left_pixels)
        right_x_vals, right_y_vals = extract_lane_pixels(right_pixels)
        
        # Adjust right_x_vals to be in the right half of the image
        right_x_vals += masked_edges.shape[1] // 2
        
        # Fit polynomials to left and right lane lines
        left_fit = fit_polynomial(left_x_vals, left_y_vals)
        right_fit = fit_polynomial(right_x_vals, right_y_vals)
        
        # Draw lane lines on the frame
        frame_with_lanes = draw_lane(frame, left_fit, right_fit)
        
        # Calculate the curvature
        y_eval = frame.shape[0]  # evaluate curvature at the bottom of the image
        left_curvature = calculate_curvature(y_eval, left_fit)
        right_curvature = calculate_curvature(y_eval, right_fit)
        
        # Calculate the average curvature
        if left_curvature is not None and right_curvature is not None:
            curvature = (left_curvature + right_curvature) / 2
            curvature_text = f"Radius of Curvature: {curvature:.2f}m"
            cv2.putText(frame_with_lanes, curvature_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        end_time = time.time()
        # Calculate FPS
        total_exec_time = end_time - start_time
        fps = 1 / (end_time - start_time)

        # Display FPS on the output image (optional)
        cv2.putText(frame_with_lanes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Processed Video', frame_with_lanes)
        
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