import json
import os
import cv2
import random
import glob
import torch
import numpy as np
from skimage import io # for io.imread
from IPython.display import clear_output
from matplotlib import pyplot as plt
from matplotlib import colors # ploting


def imshow(images, titles, nrows = 0, ncols=0, figsize = (15,20)):
    """Plot a multiple images with titles.

    Parameters
    ----------
    images : image list
    titles : title list
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    if ncols == 0 and nrows == 0:
      ncols = len(images)
      nrows = 1
    if ncols == 0:
      ncols = len(images) // nrows
    if nrows == 0:
      nrows = len(images) // ncols

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize = figsize)
    for i, image in enumerate(images):
        axeslist.ravel()[i].imshow(image, cmap=plt.gray(), vmin=0, vmax=255)
        axeslist.ravel()[i].set_title(titles[i])
        axeslist.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


if not os.path.exists('./images'):
    os.mkdir('./images')
else:
    print('zaebis')

image_dict = {}

for filename in sorted(glob.glob('./images/*.jpeg')):
    name = os.path.basename(filename)
    prefix = name.split("_")[0]

    img = io.imread(filename)

    if prefix in image_dict:
        image_dict[prefix].append((img))
    else:
        image_dict[prefix] = [(img)]

image1, image2 = image_dict['example2']
# imshow( [image1, image2], ['Left', 'Right'])
print(f'Rasmery: {image1.shape, image2.shape}')

X_shift = 100
Y_shift = 100
tx = 114
ty = 136

assert tx + X_shift >= 0
assert ty + Y_shift >= 0

size = (image1.shape[0] + image2.shape[0], image1.shape[1] + image2.shape[1], 3)
image_trans = np.uint8(np.zeros(size))

# put image 1 on resulting image
image_trans[Y_shift:Y_shift+image1.shape[0], X_shift:X_shift+image1.shape[1], :] = image1

# put image 2 on resulting image
image_trans[Y_shift+ty:Y_shift+ty+image2.shape[0], X_shift+tx:X_shift+tx+image2.shape[1], :] = image2

image_trans[:, X_shift+tx, :] = [255, 0, 0]
image_trans[Y_shift+ty, :, :] = [0, 255, 0]

# imshow( [image_trans], ['Translation-based panorama'])


# with open('manual_panorama.json', 'w') as iofile:
#     json.dump([tx, ty], iofile)

def extract_key_points(img1, img2):
    sift = cv2.SIFT_create()
    kpts1, desc1 = sift.detectAndCompute(img1, None)
    kpts2, desc2 = sift.detectAndCompute(img2, None)
    return kpts1, desc1, kpts2, desc2

kp1, des1, kp2, des2 = extract_key_points(image1, image2)
print("Coordinates of the first keypoint of image1: ", kp1[0].pt)
print("Descriptor of the first keypoint of image1:\n", des1[3354])
print()
print("Descriptor of the first keypoint of image1:\n", des2[3642])
print((np.sqrt(np.sum((des1[3354] - des2[3642]) ** 2))))


# do not change the code in the block below
# __________start of block__________
with open('keypoints_sift.json', 'r') as f:
    loaded_data = json.load(f)

for kp, loaded_kp in zip(kp1[:10], loaded_data['keypoints1']):
    assert np.allclose(kp.pt, loaded_kp, atol=1e-5), f"keypoint {kp.pt} and {loaded_kp} are not close"
# __________end of block__________


def match_key_points_cv(des1, des2):
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)

    sorted_matches = sorted(matches, key=lambda x: x.distance)
    return sorted_matches


def showMatches(img1, kp1, img2, kp2, matches, name):
    img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    imshow([img], [name])


matches_cv = match_key_points_cv(des1, des2)

print(matches_cv[0].queryIdx, matches_cv[0].trainIdx, matches_cv[0].distance)
# showMatches(image1,kp1,image2,kp2,matches_cv,"all matches")
# showMatches(image1,kp1,image2,kp2,matches_cv[:50],"best 50 matches")

class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    # Compute pairwise distances using broadcasting
    dist_matrix = np.sqrt(np.sum((des1[:, np.newaxis] - des2) ** 2, axis=2))

    # Find best matches in both directions
    best_for_des1 = np.argmin(dist_matrix, axis=1)
    best_for_des2 = np.argmin(dist_matrix, axis=0)

    # Create indices arrays for cross-check
    query_indices = np.arange(des1.shape[0])
    train_indices = np.arange(des2.shape[0])

    # Find mutual matches using boolean indexing
    mask = best_for_des2[best_for_des1] == query_indices
    mutual_matches = query_indices[mask]

    # Create DummyMatch objects
    matches = [
        DummyMatch(
            i,
            best_for_des1[i],
            dist_matrix[i, best_for_des1[i]]
        )
        for i in mutual_matches
    ]

    return sorted(matches, key=lambda x: x.distance)


def test_numpy_bf_matcher_equivalence(des1, des2):
    # OpenCV BFMatcher
    cv_matches = match_key_points_cv(des1, des2)

    # Our matcher
    np_matches = match_key_points_numpy(des1, des2)

    # Compare match indices and distances
    assert len(cv_matches) == len(np_matches), f"Match count mismatch: {len(cv_matches)} vs {len(np_matches)}"

    for idx, (m_cv, m_np) in enumerate(zip(cv_matches, np_matches)):
        assert m_cv.queryIdx == m_np.queryIdx
        assert m_cv.trainIdx == m_np.trainIdx
        assert abs(
            m_cv.distance - m_np.distance) < 1e-4, f"Distance mismatch on {idx}th match: {m_cv.distance:.4f} vs {m_np.distance:.4f}"

    print("Your numpy implementation matches OpenCV BFMatcher output!")


# test_numpy_bf_matcher_equivalence(des1, des2)

def findHomography_dlt_opencv(matches, keypoint1, keypoint2, mode='DLT'):

    src_pts = np.float32([keypoint1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([keypoint2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    if mode == 'DLT':
        mode = 0
    elif mode == 'RANSAC':
        mode = cv2.RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, mode)
    mask = mask.ravel().tolist()

    inliers = []
    for i in range(len(mask)):
      if mask[i] == 1:
        inliers.append(matches[i])

    return H, inliers


H_for_panorama, inliers = findHomography_dlt_opencv(matches_cv, kp1, kp2, 'RANSAC')
# showMatches(image1,kp1,image2,kp2,inliers,"inliers only, RANSAC")


H, inliers = findHomography_dlt_opencv(matches_cv, kp1, kp2, 'DLT')
# showMatches(image1,kp1,image2,kp2,inliers,"DLT, all matches")

H, inliers = findHomography_dlt_opencv(matches_cv[:50], kp1, kp2, 'DLT')
# showMatches(image1,kp1,image2,kp2,inliers,"DLT, top 50 matches")

def panorama(img1, img2, H, size):
    img = np.uint8(np.zeros(size))
    img = cv2.warpPerspective(src=img1, dst=img, M=np.eye(3), dsize=(size[1], size[0]))
    img = cv2.warpPerspective(src=img2, dst=img, M=H, dsize=(size[1], size[0]), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT)

    return img

size = (1280, 960*2, 3)
# imshow([panorama(image1, image2, H_for_panorama, size)],["Panorama"])

def panorama_pipeline(img1, img2, size):
    kp1, des1, kp2, des2 = extract_key_points(img1, img2)
    matches_cv = match_key_points_cv(des1, des2)
    H, inliers = findHomography_dlt_opencv(matches_cv[:50], kp1, kp2, 'RANSAC')
    res = panorama(img1, img2, H, size)

    return res, H

size = (1280, 960*2, 3)
h_dict = {}

for filename, (img1, img2) in image_dict.items():
    final_image, H = panorama_pipeline(img1, img2, size)
    h_dict[filename] = H.tolist()
    imshow([final_image],[filename])

with open('h_submission_dict.json', 'w') as iofile:
    json.dump(h_dict, iofile)