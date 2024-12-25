import math
import cv2
import numpy as np
import handTrackingModule as htm
import time
import tkinter as tk
from tkinter import filedialog

#######################
brushThickness = 10
eraserThickness = 30
selectedThickness = brushThickness
maxThickness = 100
minThickness = 1
rgbDrawColor = (1, 1, 1)  # near-black
bgrDrawColor = rgbDrawColor[::-1]
eraseColor = (0, 0, 0)
cameraWidth = 640  # Original camera dimensions for computation
cameraHeight = 480
monitorWidth = 1920  # Monitor dimensions for display
monitorHeight = 1080
screen_width = 640
screen_height = 480
eraser_mode = False
hueEnabled = False
colorPicking = False
was_in_hue = False
was_in_resize = False

root = tk.Tk()
root.withdraw()

########################


def is_finger_on_brush(indexX, indexY):
    return indexX < screen_width * 0.25 and indexY > screen_height * 0.9


def is_finger_on_erase(indexX, indexY):
    return indexX > screen_width * 0.25 and indexX < screen_width * 0.5 and indexY > screen_height * 0.9


def is_finger_on_resize(indexX, indexY):
    return indexX > screen_width * 0.50 and indexX < screen_width * 0.75 and indexY > screen_height * 0.9


def is_finger_on_hue(index_x, index_y):
    return index_x > screen_width * 0.75 and index_y > screen_height * 0.9


def save_image(img):
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    black = np.all(img_bgra[:, :, :3] == [0, 0, 0], axis=-1)
    img_bgra[black, 3] = 0
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[
                                             ("PNG files", "*.png"), ("All files", "*.*")])
    if file_path:
        cv2.imwrite(file_path, img_bgra)
        print(f"Image saved to {file_path}")
    else:
        print("Save operation cancelled")


def main():
    global was_in_resize, colorPicking, hueEnabled, brushThickness, eraserThickness, selectedThickness, rgbDrawColor, bgrDrawColor
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Turn your webcam on!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cameraWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cameraHeight)
    prev_time = 0
    curr_time = 0
    detector = htm.handDetector(maxHands=1, detectionConfidence=0.85)
    xp, yp = 0, 0
    imgCanvas = np.zeros((cameraHeight, cameraWidth, 3),
                         np.uint8)  # Blank canvas
    drawing = False
    brushing = True  # Initialize brushing state
    erasing = False  # Initialize erasing state
    resizing = False
    fill = False
    hueEnabled = False

    # Resize the image and canvas to the monitor size for display
    scaling_factor = (monitorHeight * .9) / cameraHeight
    new_width = int(cameraWidth * scaling_factor)
    new_height = int(cameraHeight * scaling_factor)

    ########################
    # Image Hue settings
    hue_image = cv2.imread("hue.png")
    if hue_image is None:
        print("Error loading hue.png")
    else:
        # Resize the hue image to fit within the camera image if needed
        hue_image = cv2.resize(
            hue_image, (new_width, new_height // 10), interpolation=cv2.INTER_AREA)
    hue_image_opacity = 1  # Adjust transparency (0.0 to 1.0)
    hue_height = hue_image.shape[0]
    hue_width = hue_image.shape[1]
    ########################
    overlay_image = cv2.imread("overlay.png")
    if overlay_image is None:
        print("Error loading overlay.png")
    else:
        # Resize the overlay image if needed
        overlay_image = cv2.resize(
            overlay_image, (new_width, new_height // 10), interpolation=cv2.INTER_AREA)
    overlay_image_opacity = 1  # Adjust transparency (0.0 to 1.0)
    overlay_height = overlay_image.shape[0]
    overlay_width = overlay_image.shape[1]

    indexX, indexY = -1, -1

    while True:
        # 1: Capture img from camera (640x480)
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # 2: Find hand landmarks in the original 640x480 frame
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        if len(landmarkList) != 0:
            # Get landmark for index finger (landmark 8)
            indexX, indexY = landmarkList[8][1:]
            # Get landmark for thumb (landmark 4)
            thumbX, thumbY = landmarkList[4][1:]

            if drawing:
                cv2.line(imgCanvas, (xp, yp), (indexX, indexY),
                         bgrDrawColor, int(selectedThickness))
            elif resizing:

                if bgrDrawColor == (0, 0, 0):
                    cv2.line(img, (indexX, indexY),
                             (thumbX, thumbY), eraseColor, 3)
                    length = math.hypot(indexX - thumbX, indexY - thumbY)
                    print("Eraser thickness: " + str(length))
                    eraserThickness = length
                    selectedThickness = eraserThickness
                else:
                    cv2.line(img, (indexX, indexY),
                             (thumbX, thumbY), (255, 255, 255), 3)
                    length = math.hypot(indexX - thumbX, indexY - thumbY)
                    print("Brush thickness: " + str(length))
                    brushThickness = length
                    selectedThickness = brushThickness
            elif colorPicking:
                selectedThickness = brushThickness
                bgrDrawColor = rgbDrawColor[::-1]

                print("This is working!")
                # Calculate the corresponding x-coordinate in the hue_image
                hueX = int((indexX / cameraWidth) * hue_width)

                # Get the color from the hue image at the specified hueX
                # Ensure that the pixel accessed is within the bounds of the hue_image
                # Check if hueX is within the width of the image
                if hueX < hue_image.shape[1]:
                    # Access the pixel value at (0, hueX) and convert it to a tuple of integers
                    bgrDrawColor = tuple(int(c)
                                         for c in hue_image[0, hueX])
                    print(f"Color picked: {bgrDrawColor}")
                    rgbDrawColor = bgrDrawColor[::-1]
                else:
                    print("hueX is out of bounds!")

            # Update the previous points for the next line segment
            xp, yp = indexX, indexY
        # else:
        #     xp, yp = 0, 0  # Reset coordinates when hand is not detected

        # Create a grayscale version of the canvas for combining
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

        # Combine the camera feed and the canvas
        # Remove drawing from the img where there is canvas drawing
        img = cv2.bitwise_and(img, imgInv)
        # Add the canvas drawing to the img
        img = cv2.bitwise_or(img, imgCanvas)

        # Draw the circle on the index finger above the combined image
        if len(landmarkList) != 0:
            index_finger_tip = landmarkList[8]
            cv2.circle(img, (indexX, indexY), int(
                selectedThickness // 2), (255, 255, 255), 2)

        # Create a larger window for drawing
        imgLarge = cv2.resize(img, (new_width, new_height),
                              interpolation=cv2.INTER_AREA)

        # # Calculate FPS
        # curr_time = time.time()
        # fps = 1 / (curr_time - prev_time)
        # prev_time = curr_time

        # # Display FPS on the resized image
        # cv2.putText(imgLarge, str(int(fps)), (10, 70),
        #             cv2.FONT_HERSHEY_PLAIN, 3, bgrDrawColor, 3)

        if hue_image is not None and hueEnabled:
            # Create an overlay, where the hue image will only span a portion of the canvas
            imgLarge[0:0 + hue_height, 0:0 + hue_width] = cv2.addWeighted(
                imgLarge[0:0 + hue_height, 0:0 + hue_width], 0, hue_image, hue_image_opacity, 0)

        if overlay_image is not None:
            start_y = imgLarge.shape[0] - overlay_height
            # Create an overlay, where the hue image will only span a portion of the canvas
            imgLarge[start_y:start_y + overlay_height, 0:0 + overlay_width] = cv2.addWeighted(
                imgLarge[start_y:start_y + overlay_height, 0:0 + overlay_width], 0, overlay_image, overlay_image_opacity, 0)

        # Add text showing the brush keybind
        text_position_x = int(new_width * 0.18)
        text_position_y = int(new_height * 0.96)
        draw_size_text = f"(B)"
        cv2.putText(imgLarge, draw_size_text, (text_position_x, text_position_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add text showing the erase keybind
        text_position_x = int(new_width * 0.43)
        text_position_y = int(new_height * 0.96)
        draw_size_text = f"(E)"
        cv2.putText(imgLarge, draw_size_text, (text_position_x, text_position_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add text showing the hue
        text_position_x = int(new_width * 0.55)
        text_position_y = int(new_height * 0.96)
        draw_size_text = f"Size: {int(selectedThickness)}   (R)"
        cv2.putText(imgLarge, draw_size_text, (text_position_x, text_position_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Create hue rectangle on the bottom right corner
        rect_width = int(.25 * new_width)
        rect_height = int(.1 * new_height)
        top_left_x = int(new_width * 0.88) - rect_width // 2
        top_left_y = int(new_height * 0.95) - rect_height // 2
        bottom_right_x = top_left_x + rect_width
        bottom_right_y = top_left_y + rect_height
        cv2.rectangle(imgLarge, (top_left_x, top_left_y),
                      (bottom_right_x, bottom_right_y), bgrDrawColor, -1)
        
        # Add text showing the draw size
        text_position_x = int(new_width * 0.83)
        text_position_y = int(new_height * 0.96)
        draw_size_text = f"Color   (H)"
        cv2.putText(imgLarge, draw_size_text, (text_position_x, text_position_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the final resized image
        cv2.imshow("Capture Image", imgLarge)
        # cv2.imshow("Canvas", imgCanvas)

        # Exit condition
        key = cv2.waitKey(1)

        if key == ord('q') or key == ord('Q'):  # Quit when 'q' is pressed
            break
        elif key == ord('c') or key == ord('C'):  # Clear the canvas when 'c' is pressed
            imgCanvas = np.zeros((cameraHeight, cameraWidth, 3), np.uint8)
            print("clear")
        elif key == 32 and not hueEnabled:  # Toggle drawing mode when spacebar is pressed
            drawing = not drawing
            resizing = False
            hueEnabled = False
            print("Drawing: " + str(drawing))
        elif key == ord('b') or key == ord('B'):
            selectedThickness = brushThickness
            bgrDrawColor = rgbDrawColor[::-1]
            print("brush thickness is: " + str(selectedThickness))
        elif is_finger_on_brush(indexX, indexY):
            selectedThickness = brushThickness
            bgrDrawColor = rgbDrawColor[::-1]
            print("brush thickness is: " + str(selectedThickness))
        # Toggle erasing mode when 'e' is pressed
        elif key == ord('e') or key == ord('E'):
            selectedThickness = eraserThickness
            bgrDrawColor = (0, 0, 0)
            print("eraser thickness is: " + str(selectedThickness))
        elif is_finger_on_erase(indexX, indexY):
            selectedThickness = eraserThickness
            bgrDrawColor = (0, 0, 0)
            print("eraser thickness is: " + str(selectedThickness))
        elif key == 13:  # Save the canvas when 'Enter' is pressed
            canvas_copy = imgCanvas.copy()
            save_image(canvas_copy)
            print("Canvas saved as 'canvas_output.png'")
        elif key == ord('r') or key == ord('R'):  # Resize brush or eraser
            print("Resize")
            resizing = not resizing
            drawing = False
            hueEnabled = False
        elif is_finger_on_resize(indexX, indexY):  # Resize brush or eraser
            if not was_in_resize:
                resizing = not resizing
                drawing = False
                hueEnabled = False
                print("Resize")
            was_in_resize = True
        elif key == ord('h') or key == ord('H'):
            print("Color pick")
            drawing = False
            resizing = False
            hueEnabled = not hueEnabled
        elif is_finger_on_hue(indexX, indexY):
            if not was_in_hue:
                hueEnabled = not hueEnabled
                drawing = False
                resizing = False
                print("Color picking mode toggled: " + str(hueEnabled))
            was_in_hue = True
        elif hueEnabled and key == 32:
            colorPicking = not colorPicking
        else:
            was_in_hue = False
            was_in_resize = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
