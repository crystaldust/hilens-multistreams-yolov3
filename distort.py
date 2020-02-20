import numpy as np
import cv2

# 镜头畸变系数
DEF_LENS_MTX = np.array([
				[1.58206604e+03, 0., 9.87620699e+02],
				[0., 1.57889682e+03, 5.43361354e+02],
				[0., 0., 1.],
		   ])

DEF_LENS_DIST = np.array([[-0.29047252, -0.1558767, 0.0004676, -0.0008854, 0.26806017]])


def lens_distortion_adjustment(image, lens_mtx=DEF_LENS_MTX, lens_dist=DEF_LENS_DIST):
	h, w = image.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(lens_mtx, lens_dist, (w, h), 0, (w, h))  # 自由比例参数
	dst = cv2.undistort(image, lens_mtx, lens_dist, None, newcameramtx)
	return dst

if __name__ == "__main__":
	# 示例， 先处理变形，后resize
	cap = cv2.VideoCapture('rtsp://192.168.0.157:554/stream1')
	ret, frame = cap.read()
	frame = lens_distortion_adjustment(frame)
	frame = cv2.resize(frame,(960, 540))
