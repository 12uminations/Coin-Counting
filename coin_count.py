import cv2
import numpy as np

def coinCounting(filename):
    im = cv2.imread(filename)

    # Convert to HSV for better color handling under varying lighting
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Resize the image for faster processing
    target_size = (int(im.shape[1] / 2), int(im.shape[0] / 2))
    hsv = cv2.resize(hsv, target_size)

    # Apply median filter to reduce noise
    hsv = cv2.medianBlur(hsv, 5)  # Kernel size of 5, can be tuned

    # Define color ranges for yellow and blue in HSV
    yellow_lower = (25, 120, 120)
    yellow_upper = (35, 255, 255)
    blue_lower = (93, 150, 100)  
    blue_upper = (130, 255, 255)

    # Create masks for yellow and blue coins
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # Define HSV ranges for light blue
    light_blue_lower = (90, 50, 150)  # Light blue with lower saturation and higher value
    light_blue_upper = (120, 150, 255)

    # Kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Apply erosion and dilation to clean up yellow mask
    mask_yellow = cv2.erode(mask_yellow, kernel, iterations=3)
    mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=4)
    mask_yellow = cv2.erode(mask_yellow, kernel, iterations=2)

    # Apply erosion and dilation to clean up blue mask
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=3)
    mask_blue = cv2.erode(mask_blue, np.ones((7, 7), np.uint8))

    mask_light_blue = cv2.inRange(hsv, light_blue_lower, light_blue_upper)
    mask_blue = cv2.bitwise_or(mask_blue, mask_light_blue)
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)
    mask_blue = cv2.erode(mask_blue, np.ones((24, 23), np.uint8))
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=4)
    mask_blue = cv2.erode(mask_blue, np.ones((9, 9), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    radii = []

    for contour in contours:
         # Find the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        if radius > 115:
            mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)
            mask_blue = cv2.erode(mask_blue, np.ones((31, 31), np.uint8))
        

    def count_contours(mask):
        # Find and count contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)

    # Count yellow and blue coins using contours
    yellow = count_contours(mask_yellow)
    blue = count_contours(mask_blue)

    # Resize output images to 500x500 for display
    im_resized = cv2.resize(im, (500, 500))
    mask_yellow_resized = cv2.resize(mask_yellow, (500, 500))
    mask_blue_resized = cv2.resize(mask_blue, (500, 500))

    cv2.imshow('Original Image', im_resized)
    cv2.imshow('Yellow Coin', mask_yellow_resized)
    cv2.imshow('Blue Coin', mask_blue_resized)
    cv2.waitKey()
    return [yellow, blue]

# Expected answers
expected_answers = np.array([[5, 8], [6, 3], [2, 4], [2, 4], [1, 7], [3, 5], [4, 3], [5, 5], [2, 6], [4, 2]])

# Path to the folder containing coin images
image_folder = './CoinCounting/'
# Evaluate each image and compare with expected answers
for i in range(1, 11):
    result = coinCounting(image_folder + 'coin' + str(i) + '.jpg')
    print(f"Image {i}: Detected - {result}")

    # Check if the result matches the expected answer
    #if result == list(expected_answers[i - 1]):
        #print(f"Image {i}: Correct\n")
    #else:
       # print(f"Image {i}: Incorrect\n")
