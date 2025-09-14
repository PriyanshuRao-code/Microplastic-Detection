import cv2
import argparse

# ---------------------------
# Global Config
# ---------------------------
SHOW_IMAGE = True  # Set to False to disable image display

# ---------------------------
#  Custom CNN (Baseline)
# ---------------------------
def run_customcnn(image_path):
    image = cv2.imread(image_path, 0)
    if image is None:
        print(f"Error: Unable to read image at path '{image_path}'")
        return 0

    _, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours = object candidates
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)
    
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, contours, -1, (0, 0, 255), 2)
    
    if SHOW_IMAGE:
        cv2.imshow("Custom CNN Approx (Contours)", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"Detected: {count}")
    return count

# ---------------------------
# Main Function
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--show", action="store_true", help="Show image output")  # Optional argument to control display
    args = parser.parse_args()

    # Set global flag based on command-line argument
    SHOW_IMAGE = args.show

    run_customcnn(args.image)
