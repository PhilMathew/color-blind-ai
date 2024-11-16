import cv2
import numpy as np

def isolate_skin(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    cv2.imwrite(output_path, skin)
    print(f"Skin-isolated image saved at: {output_path}")

#"C:\Users\Shuya Shou\Downloads\IMG_4237.jpg"
#Documents/fall24/cs787/isolate.py
image_path = 'IMG_4237.jpg'
output_path = './output_IMG_4237.jpg'
isolate_skin(image_path, output_path)
