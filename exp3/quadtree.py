import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("img.jpg")

if img is None:
    print("Error: Image not found")
    exit()

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------- SPLIT IMAGE --------
def split4(image):
    h, w = image.shape[:2]
    h_mid = h // 2
    w_mid = w // 2

    north_west = image[:h_mid, :w_mid]
    north_east = image[:h_mid, w_mid:]
    south_west = image[h_mid:, :w_mid]
    south_east = image[h_mid:, w_mid:]

    return north_west, north_east, south_west, south_east


split_img = split4(img)

# Display split images
plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1)
plt.imshow(split_img[0])
plt.title("North West")

plt.subplot(2, 2, 2)
plt.imshow(split_img[1])
plt.title("North East")

plt.subplot(2, 2, 3)
plt.imshow(split_img[2])
plt.title("South West")

plt.subplot(2, 2, 4)
plt.imshow(split_img[3])
plt.title("South East")

plt.tight_layout()
plt.show()

# -------- CONCATENATE IMAGE --------
def concatenate4(nw, ne, sw, se):
    top = np.concatenate((nw, ne), axis=1)
    bottom = np.concatenate((sw, se), axis=1)
    return np.concatenate((top, bottom), axis=0)


full_img = concatenate4(
    split_img[0],
    split_img[1],
    split_img[2],
    split_img[3]
)

# Display reconstructed image
plt.figure()
plt.imshow(full_img)
plt.title("Reconstructed Image")
plt.axis("off")
plt.show()

