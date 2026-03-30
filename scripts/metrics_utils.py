import numpy as np

def compute_hypervolume(points: np.ndarray, ref_point: tuple) -> float:
    """
    points: array of shape (N, 2), sorted or unsorted
            points[:,0] = owner utility (maximize)
            points[:,1] = fairness cost (minimize)
    ref_point: (x_ref, y_ref) = common nadir point
    """
    pts = points[np.argsort(points[:, 1])[::-1]] # sort by y descending
    hv = 0.0

    for i in range(len(pts)):
        x_i, y_i = pts[i]
        y_prev = ref_point[1] if i == 0 else pts[i - 1][1]

        width = x_i - ref_point[0]
        height = y_prev - y_i
        if width >= 0 and height >= 0:
            hv += width * height
        else:
            raise ValueError("negative volume, order is wrong!")
    return hv

def compute_hypervolume_rawls(points: np.ndarray, ref_point: tuple) -> float:
    """
    points: array of shape (N, 2), sorted or unsorted
            points[:,0] = owner utility (maximize)
            points[:,1] = worst case fairness (maximize)
    ref_point: (x_ref, y_ref) = common nadir point
    """
    pts = points[np.argsort(points[:, 1])] # Sort by y ascending
    hv = 0.0

    for i in range(len(pts)):
        x_i, y_i = pts[i]
        y_prev = ref_point[1] if i == 0 else pts[i - 1][1]

        width = x_i - ref_point[0]
        height = y_i - y_prev
        if width >= 0 and height >= 0:
            hv += width * height
        else:
            raise ValueError("negative volume, order is wrong!")

    return hv



def interpolate_pf(pf: np.ndarray, x_grid: np.ndarray):
    pf = pf[np.argsort(pf[:, 0])]
    return np.interp(x_grid, pf[:, 0], pf[:, 1])


def stochastic_gain(det_pf, stoch_pf, n_grid=200, fairness_type: str="egal"):
    """
    Gain = det_y - stoch_y
    Positive => stochastic improves fairness
    """
    x_min = max(det_pf[:, 0].min(), stoch_pf[:, 0].min())
    x_max = min(det_pf[:, 0].max(), stoch_pf[:, 0].max())

    x_grid = np.linspace(x_min, x_max, n_grid)

    det_y = interpolate_pf(det_pf, x_grid)
    stoch_y = interpolate_pf(stoch_pf, x_grid)

    gain = det_y - stoch_y
    if fairness_type=="rawls":
        gain = -1*gain
    else:
        assert fairness_type=="egal"
    return x_grid, gain


def average_stochastic_gain(det_pf, stoch_pf,fairness_type: str):
    _, gain = stochastic_gain(det_pf, stoch_pf,fairness_type=fairness_type)
    return float(gain.mean())
