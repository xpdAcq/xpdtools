from xpdtools.calib4 import findrings
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt


points_image, center_pt=findrings(imarray)

new_img = np.empty((imarray.shape[0], imarray.shape[1], 4))
    max_img = np.amax(imarray)
    min_img = np.amin(imarray)
    for i, row in enumerate(new_img):
        for j, elem in enumerate(row):
            old_val = imarray[i][j]
            scaled = (old_val - min_img) / (max_img - min_img)
            new_img[i][j] = np.array(
                [1.0 - scaled, 1.0 - scaled, 1.0 - scaled, 1.0]
            )

    for point_image in points_image:
        for i in range(-2, 2):
            for j in range(-2, 2):
                new_img[point_image[0] + i][point_image[1] + j] = np.array(
                    [1.0, 0.0, 0.0, 1.0]
                )

    for i in range(-2, 2):
        for j in range(-2, 2):
            new_img[center_pt[0] + i][center_pt[1] + j] = np.array(
                [1.0, 0.0, 0.0, 1.0]
            )

    plt.imshow(new_img)
    plt.show()
