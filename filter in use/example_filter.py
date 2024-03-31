# press ESC to exit the demo!
from pfilter import (
    ParticleFilter,
    gaussian_noise,
    cauchy_noise,
    squared_error,
    independent_sample,
)
import numpy as numpy

# testing only
from scipy.stats import norm, gamma, uniform
import skimage.draw
import cv2


imageSize = 48


def blob(x):
    """Given an Nx3 matrix of blob pt and size,
    create N imageSize x imageSize images, each with a blob drawn on
    them given by the value in each row of x

    One row of x = [x,y,radius]."""
    y = numpy.zeros((x.shape[0], imageSize, imageSize))
    for i, particle in enumerate(x):
        rr, cc = skimage.draw.circle_perimeter(
            int(particle[0]), int(particle[1]), int(max(particle[2], 1)), shape=(imageSize, imageSize)
        )
        y[i, rr, cc] = 1
    return y


#%%

# names (this is just for reference for the moment!)
columns = ["x", "y", "radius", "dx", "dy"]


# prior sampling function for each variable
# (astotes x and y are coordinates in the range 0-imageSize)
# gamma(a=1, loc=0, sl=10).rvs,
prior_fn = independent_sample(
    [
        norm(loc=imageSize / 2, scale=imageSize / 2).rvs,
        norm(loc=imageSize / 2, scale=imageSize / 2).rvs,
        gamma(a=1, loc=0, scale=10).rvs,
        norm(loc=0, scale=0.5).rvs,
        norm(loc=0, scale=0.5).rvs,
    ]
)

# very simple linear dynamics: x += dx
def velocity(x):
    dt = 1.0
    xp = (
        x
        @numpy.array(
            [
                [1, 0, 0, dt, 0],
                [0, 1, 0, 0, dt],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ).T
    )

    return xp

#

# n_particles=100
def example_filter():
    # create the filter
    pf = ParticleFilter(
        prior_fn=prior_fn,
        observe_fn=blob,
        n_particles=100,
        dynamics_fn=velocity,
        noise_fn=lambda x: cauchy_noise(x, sigmas=[0.05, 0.05, 0.01, 0.005, 0.005]),
        weight_fn=lambda x, y: squared_error(x, y, sigma=2),
        s_proportion=0.2,
        column_names=columns,
    )

    # numpy.random.seed(2018)
    # start in centre, random radius
    s = numpy.random.uniform(2, 8)

    # random movement direction
    dx = numpy.random.uniform(-0.25, 0.25)
    dy = numpy.random.uniform(-0.25, 0.25)

    # appear at centre
    x = imageSize // 2
    y = imageSize // 2
    sl_fa = 20

    # create window
    cv2.namedWindow("samples", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("samples", sl_fa * imageSize, sl_fa * imageSize)

    for i in range(1000):

        # generate the actual image
        low_res_image = blob(numpy.array([[x, y, s]]))
        pf.update(low_res_image)

        # resize for drawing onto
        image = cv2.resize(
            numpy.squeeze(low_res_image), (0, 0), fx=sl_fa, fy=sl_fa
        )

        cv2.putText(
            image,
            "ESC to exit",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        color = cv2.cvtColor(image.astype(numpy.float32), cv2.COLOR_GRAY2RGB)

        x_h, y_h, s_h, dx_h, dy_h = pf.mean_state

        # draw individual particles
        for particle in pf.original_particles:

            xa, ya, sa, _, _ = particle
            sa = numpy.clip(sa, 1, 100)
            cv2.circle(
                color,
                (int(ya * sl_fa), int(xa * sl_fa)),
                max(int(sa * sl_fa), 1),
                (1, 0, 0),
                1,
            )

        # x,y exchange because of ordering between skimage and opencv
        cv2.circle(
            color,
            (int(y_h * sl_fa), int(x_h * sl_fa)),
            max(int(sa * sl_fa), 1),
            (0, 1, 0),
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            color,
            (int(y_h * sl_fa), int(x_h * sl_fa)),
            (
                int(y_h * sl_fa + 5 * dy_h * sl_fa),
                int(x_h * sl_fa + 5 * dx_h * sl_fa),
            ),
            (0, 0, 1),
            lineType=cv2.LINE_AA,
        )

        cv2.imshow("samples", color)
        result = cv2.waitKey(20)
        # break on escape
        if result == 27:
            break
        x += dx
        y += dy
        input("Press Enter to continue...")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    example_filter()
