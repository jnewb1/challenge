import pathlib
import cv2
import numpy as np

class ChallengeVideo:
    def __init__(self, base_path: pathlib.Path):
        vid = cv2.VideoCapture(str(base_path.with_suffix(".hevc")))

        self.frames = []
        while True:
            ret, frame = vid.read()
            if ret:
                self.frames.append(frame)
            else:
                break

        self.labels = np.array([(float(x.split(" ")[0]), float(x.split(" ")[1])) for x in open(str(base_path.with_suffix(".txt")), "r").readlines()])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, slice):
        return self.frames[slice], self.labels[slice]


if __name__ == "__main__":
    c = ChallengeVideo(pathlib.Path("calib_challenge/labeled/1"))

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(2,2)
    fig.set_size_inches(10, 10)

    def plot_frame(ax, index):
        ax.set_title(f"Pitch: {np.rad2deg(c[index][1][0]):.2f}\nYaw: {np.rad2deg(c[index][1][1]):.2f}")
        ax.imshow(c[index][0])
    
    plot_frame(axes[0,0], 0)
    plot_frame(axes[0,1], 1)
    plot_frame(axes[1,0], 2)
    plot_frame(axes[1,1], 3)

    plt.savefig("dataset.png")