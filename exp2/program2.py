import cv2
import numpy as np

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

# Usage
image = cv2.imread("img.jpg")
angle_degrees = 45
rotated = rotate_image(image, angle_degrees)
cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()


def scale_image(image, scale_x, scale_y):
    scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
    return scaled_image

# Usage
image = cv2.imread("img.jpg")
scale_factor_x = 1.5
scale_factor_y = 1.5
scaled = scale_image(image, scale_factor_x, scale_factor_y)
cv2.imshow("Scaled Image", scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()


def skew_image(image, skew_x, skew_y):
    height, width = image.shape[:2]
    skew_matrix = np.float32([[1, skew_x, 0],
                              [skew_y, 1, 0]])
    skewed_image = cv2.warpAffine(image, skew_matrix, (width, height))
    return skewed_image

# Usage
image = cv2.imread("img.jpg")
skew_factor_x = 0.2
skew_factor_y = 0.1
skewed = skew_image(image, skew_factor_x, skew_factor_y)
cv2.imshow("Skewed Image", skewed)
cv2.waitKey(0)
cv2.destroyAllWindows()


def affine_transform(image, pts_src, pts_dst):
    matrix = cv2.getAffineTransform(pts_src, pts_dst)
    transformed_image = cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0])
    )
    return transformed_image

image = cv2.imread("img.jpg")
src_points = np.float32([[50, 50], [200, 50], [50, 200]])
dst_points = np.float32([[10, 100], [200, 50], [100, 250]])
affine_transformed = affine_transform(image, src_points, dst_points)
cv2.imshow("Affine Transformed Image", affine_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()


def bilinear_transform(image, pts_src, pts_dst):
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    transformed_image = cv2.warpPerspective(
        image, matrix, (image.shape[1], image.shape[0])
    )
    return transformed_image

image = cv2.imread("img.jpg")
src_points = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
dst_points = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
bilinear_transformed = bilinear_transform(image, src_points, dst_points)

cv2.imshow("Bilinear Transformed Image", bilinear_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
