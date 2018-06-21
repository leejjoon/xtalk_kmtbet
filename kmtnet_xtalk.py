import numpy as np
import scipy.ndimage as ni

from astropy.modeling import models, fitting


def estimate_saturation_level(hdu_list, data_logger=None):
    bins = np.arange(30000, 89000, 400)

    thresh_list = []

    for i, d in enumerate(hdu_list):
        h, _ = np.histogram(np.ravel(d.data[d.data > 30000]),
                            bins=bins)
        imax = np.argmax(h)

        x0 = bins[imax]

        g_init = models.Gaussian1D(amplitude=1., mean=x0, stddev=1000.)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, bins[:-1], h)

        thresh = g.parameters[1] - 5*np.abs(g.parameters[2])
        thresh_list.append(thresh)

        if data_logger is not None:
            title = "estimate_saturation_level"
            log_data = dict(bins=bins, hist=h,
                            model=g(bins),
                            model_params=g.parameters,
                            thresh=thresh)
            data_logger.log(title=title,
                            log_id=i,
                            log_data=log_data)

    return thresh_list


class Plot_estimate_saturation_level():
    def __init__(self, fig=None, nrows_ncols=(8, 4)):
        
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()

        from mpl_toolkits.axes_grid1 import Grid

        self.grid = Grid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                         axes_pad=0.1, share_y=False,
                         direction="column")  # pad between axes in inch.

    def log(self, title, log_id, log_data):
        ax = self.grid[log_id]

        bins = log_data["bins"]
        h = log_data["hist"]
        model = log_data["model"]
        thresh = log_data["thresh"]

        ax.fill_between(bins[:-1], h, color="0.8")
        ax.plot(bins, model, "r-")

        ax.axvline(thresh, color="k")

        ax.set_ylim(0, 1.2*log_data["model_params"][0])
        ax.set_xlim(35000, 89000)


def get_flip_roll(hdu_list, source_i, victim_i, thresh_list,
                  fig=None, subplot_spec="111",
                  return_coeff=False):

    datasec_slice = (slice(None, None), slice(None, 1152))

    source_data = hdu_list[source_i].data[datasec_slice]
    victim_data = hdu_list[victim_i].data[datasec_slice]

    bg = np.array([np.median(row[row < 30000]) for row in victim_data])

    mef3data = victim_data - bg[:, np.newaxis]

    msk = ni.binary_erosion(source_data > 10000)
    msk = msk & ~(source_data > thresh_list[source_i])

    msk_satu = ni.binary_erosion(source_data > thresh_list[source_i])

    #msk = msk1

    cc_list1 = []
    cc_list0 = []

    dx = source_data[msk]
    dxs = source_data[msk_satu]

    mmm = 0
    ratio = []

    # mef3data1 = mef3data[:, ::-1]
    flipped_data = {True: mef3data[:, ::-1], False: mef3data}

    flip_roll_candidates = [(False, -1),
                            (False, 0),
                            (False, 1),
                            (True, -1),
                            (True, 0),
                            (True, 1)]

    if fig is not None:
        # fig = plt.figure(1, (4., 4.))
        # fig.clf()

        from mpl_toolkits.axes_grid1 import Grid
        grid = Grid(fig, subplot_spec,  # similar to subplot(111)
                    nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                    axes_pad=0.1,
                    share_all=True)  # pad between axes in inch.
        # )
    else:
        grid = [None] * 6

    for ax, (flip, roll) in zip(grid, flip_roll_candidates):
        mef3data = flipped_data[flip]
        dy = np.roll(mef3data, roll)[msk]
        dys = np.roll(mef3data, roll)[msk_satu]

        dydx = dy / dx
        cut_upper = np.percentile(dydx, 80)
        cut_lower = np.percentile(dydx, 10)

        cut_msk = (cut_lower < dydx) & (dydx < cut_upper)
        ratio.append(np.median(dydx[cut_msk]))

        cc_list0.append(np.corrcoef(dx[cut_msk], dy[cut_msk])[0][1])
        cc_list1.append(np.corrcoef(dx, dy/dx)[0][1])

        mmm = max(mmm, np.percentile(dys, 80))

        if ax is not None:
            ax.plot(dx, dy, ".")
            ax.plot(dxs, dys, ".")

            ann_text = "flip={}, roll={}".format(flip, roll)
            ax.annotate(ann_text, (0, 1), xycoords="axes fraction",
                        xytext=(5, -5), textcoords="offset pixels",
                        ha="left", va="top")

    ii = np.argmax(np.abs(cc_list0))
    xx = np.array([10000, 60000])

    if ax is not None:
        ax.set_ylim(-100, mmm+100)
        grid[ii].plot(xx, xx*ratio[ii], "k-", lw=4)

    if return_coeff:
        return flip_roll_candidates[ii], ratio[ii]  # ratio[ii]
    else:
        return flip_roll_candidates[ii]


def flip_roll_source(source_image, flip, roll):

    source_image = np.roll(source_image, -roll)

    if flip:
        source_image = source_image[:, ::-1]

    return source_image


def flip_roll_victim(victim_image, flip, roll):
    if flip:
        victim_image = victim_image[:, ::-1]

    victim_image = np.roll(victim_image, roll)

    return victim_image


def get_coeff(hdu_list, source_i, victim_i, thresh_list,
              flip_roll):

    datasec_slice = (slice(None, None), slice(None, 1152))

    source_data = hdu_list[source_i].data[datasec_slice]
    victim_data = hdu_list[victim_i].data[datasec_slice]

    bg = np.array([np.median(row[row < 30000]) for row in victim_data])

    mef3data = victim_data - bg[:, np.newaxis]

    # msk = ni.binary_erosion(source_data > 10000)
    msk = ni.binary_erosion(source_data > 2000)
    msk = msk & ~(source_data > thresh_list[source_i])

    dx = source_data[msk]

    flip, roll = flip_roll

    dy = flip_roll_victim(mef3data, flip, roll)[msk]

    dydx = dy / dx
    cut_upper = np.percentile(dydx, 90)
    cut_lower = np.percentile(dydx, 10)

    cut_msk = (cut_lower < dydx) & (dydx < cut_upper)
    ratio = np.median(dydx[cut_msk])

    return ratio


flip_roll_default_dict = {(0, 2): (False, -1),
                          (0, 4): (True, -1),
                          (0, 6): (True, -1),
                          (1, 3): (False, -1),
                          (1, 5): (True, -1),
                          (1, 7): (True, -1),
                          (2, 0): (False, -1),
                          (2, 4): (True, -1),
                          (2, 6): (True, -1),
                          (3, 1): (False, -1),
                          (3, 5): (True, -1),
                          (3, 7): (True, -1),
                          (4, 0): (True, 1),
                          (4, 2): (True, 1),
                          (4, 6): (False, 1),
                          (5, 1): (True, 1),
                          (5, 3): (True, 1),
                          (5, 7): (False, 1),
                          (6, 0): (True, 1),
                          (6, 2): (True, 1),
                          (6, 4): (False, 1),
                          (7, 1): (True, 1),
                          (7, 3): (True, 1),
                          (7, 5): (False, 1)}


def get_default_flip_roll_dict(offsets=None):

    if offsets is None:
        offsets = [0, 8, 16, 24]

    keys = flip_roll_default_dict.keys()
    for offset in offsets:
        for source_i, victim_i in keys:
            v = flip_roll_default_dict[(source_i, victim_i)]
            source_i = source_i + offset
            victim_i = victim_i + offset
            flip_roll_default_dict[(source_i, victim_i)] = v

    return flip_roll_default_dict


def find_flip_roll_dict(hdu_list, thresh_list, offsets=None,
                        sv_list=None,
                        return_coeff=False):

    flip_roll_dict = {}
    coeff_dict = {}

    if offsets is None:
        offsets = [0, 8, 16, 24]

    if sv_list is None:
        sv_list = [(i, j) for i in range(8) for j in range(8)
                   if (i != j) and ((i - j) % 2 == 0)]

    for offset in offsets:
        # sv_list = [(0, 2), (0, 4), (0, 6), 
        #            (2, 0), (2, 4), (2, 6),
        #            (4, 0), (4, 2), (4, 6),
        #            (6, 0), (6, 2), (6, 4),
        #            (1, 3), (1, 5), (1, 7),
        #            (3, 1), (3, 5), (3, 7),
        #            (5, 1), (5, 3), (5, 7),
        #            (7, 1), (7, 3), (7, 5)]

        for source_i, victim_i in sv_list:
            source_i = source_i + offset
            victim_i = victim_i + offset

            flip_roll = get_flip_roll(hdu_list, source_i, victim_i,
                                      thresh_list,
                                      fig=None, subplot_spec="111",
                                      return_coeff=return_coeff)

            flip_roll_dict[(source_i, victim_i)] = flip_roll

    return flip_roll_dict


def get_xtalk_coeff(hdu_list, thresh_list=None, flip_roll_dict=None,
                    return_df=False):
    """
    flip_roll_dict : None, 'auto' or dict
    """
    if thresh_list is None:
        thresh_list = estimate_saturation_level(hdu_list)

    if flip_roll_dict is None:
        flip_roll_dict = get_default_flip_roll_dict()

    elif flip_roll_dict == "auto":
        flip_roll_dict = find_flip_roll_dict(hdu_list,
                                             thresh_list=thresh_list)

    ratio_dict = {}

    for source_i, victim_i in flip_roll_dict.keys():

        flip_roll = flip_roll_dict[(source_i, victim_i)]
        ratio = get_coeff(hdu_list, source_i, victim_i,
                          thresh_list, flip_roll)
        ratio_dict[(source_i, victim_i)] = ratio

    if return_df:
        return convert_to_dataframe(ratio_dict, flip_roll_dict)
    else:
        return ratio_dict, flip_roll_dict


def get_xtalk_image(hdu_list, victim_i, df_xtalk, thresh=None):
    xtalk_im = np.zeros_like(hdu_list[victim_i].data)

    datasec_slice = (slice(None, None), slice(None, 1152))

    for (si, vi), row in df_xtalk.set_index(["sou", "vic"]).iterrows():
        if vi != victim_i:
            continue
        # print si

        im = flip_roll_source(hdu_list[si].data[datasec_slice],
                              row.flip, row.roll)
        xtalk_im[datasec_slice] += im * row.coeff

    if thresh is not None:
        msk = np.abs(xtalk_im) < thresh
        msk = ni.binary_erosion(msk, iterations=2)
        xtalk_im[msk] = 0

    return xtalk_im


def convert_to_dataframe(ratio_dict, flip_roll_dict):
    import pandas as pd

    keys = sorted(ratio_dict.keys())

    # index = pd.Series(data=keys, name="source_victim")
    r = dict(sou=[k[0] for k in keys],
             vic=[k[1] for k in keys],
             flip=[flip_roll_dict[k][0] for k in keys],
             roll=[flip_roll_dict[k][1] for k in keys],
             coeff=[ratio_dict[k] for k in keys],
             distance=[np.abs(s - v) for (s, v) in keys])

    df = pd.DataFrame(r)  # , index=index)
    return df  # .set_index(["sou", "vic"])
