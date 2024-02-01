# ########################################################################### #
#    Copyright (c) 2019-2020, California Institute of Technology.
#    All rights reserved.  Based on Government Sponsored Research under
#    contracts NNN12AA01C, NAS7-1407 and/or NAS7-03001.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions
#    are met:
#      1. Redistributions of source code must retain the above copyright
#         notice, this list of conditions and the following disclaimer.
#      2. Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimer in
#         the documentation and/or other materials provided with the
#         distribution.
#      3. Neither the name of the California Institute of
#         Technology (Caltech), its operating division the Jet Propulsion
#         Laboratory (JPL), the National Aeronautics and Space
#         Administration (NASA), nor the names of its contributors may be
#         used to endorse or promote products derived from this software
#         without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CALIFORNIA
#    INSTITUTE OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ########################################################################### #
#    EXOplanet Transit Interpretation Code (EXOTIC)
#    # NOTE: See companion file version.py for version info.
# ########################################################################### #
# Exoplanet light curve analysis
#
# Fit an exoplanet transit model to time series data.
# ########################################################################### #
from astropy.time import Time
import copy
from itertools import cycle
import matplotlib.pyplot as plt
from numba import jit, njit, prange
try:
    import numpy as np
    #if 'np' in globals():
    #    del globals()['np']
    #import cupy as np
    #import torch
    import jax
    import jax.numpy as jnp
    import jax.scipy
    from jax.config import config; config.update('jax_enable_x64', True)
    #from functools import partial
    #from pylightcurve.models.exoplanet_lc import transit as pytransit
    #from pylightcurve_torch.functional import transit as pytransit
except ImportError:
    import numpy as np
    #from pylightcurve.models.exoplanet_lc import transit as pytransit
from scipy import spatial
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
try:
    from ultranest import ReactiveNestedSampler
    #import ultranest.stepsampler
    if 'jax' in globals():
        import ultranest.popstepsampler
    else:
        import ultranest.stepsampler
except ImportError:
    import dynesty
    import dynesty.plotting
    from dynesty.utils import resample_equal
    from scipy.stats import gaussian_kde

try:
    from plotting import corner
except ImportError:
    from .plotting import corner

def maybe_decorate(decorator_false):
    try:
        return jax.jit
    except NameError:
        return decorator_false

@maybe_decorate(jit(nopython=True, parallel=True, cache=True))
def weightedflux(flux, gw, nearest): # assuming only cupy arrays, if GPU
    try:
        return jnp.sum(flux[nearest] * gw, axis=-1)
    except NameError:
        return np.sum(flux[nearest] * gw, axis=-1)

@maybe_decorate(jit(nopython=True, parallel=True, cache=True))
def gaussian_weights(X, w=1, neighbors=50, feature_scale=1000): # assuming only cupy arrays, if GPU
    try:
        Xm = (X - jnp.median(X, 0)) * w
        kdtree = jax.scipy.spatial.cKDTree(Xm * feature_scale)
        nearest = jnp.zeros((X.shape[0], neighbors))
        gw = jnp.zeros((X.shape[0], neighbors), dtype=float)
        for point in range(X.shape[0]):
            ind = kdtree.query(kdtree.data[point], neighbors + 1)[1][1:]
            dX = Xm[ind] - Xm[point]
            Xstd = jnp.std(dX, 0)
            gX = jnp.exp(-dX ** 2 / (2 * Xstd ** 2))
            gwX = jnp.product(gX, 1)
            gw[point, :] = gwX / gwX.sum()
            nearest[point, :] = ind
        gw[jnp.isnan(gw)] = 0.01
    except NameError:
        Xm = (X - np.median(X, 0)) * w
        kdtree = spatial.cKDTree(Xm * feature_scale)
        nearest = np.zeros((X.shape[0], neighbors))
        gw = np.zeros((X.shape[0], neighbors), dtype=float)
        for point in range(X.shape[0]):
            ind = kdtree.query(kdtree.data[point], neighbors + 1)[1][1:]
            dX = Xm[ind] - Xm[point]
            Xstd = np.std(dX, 0)
            gX = np.exp(-dX ** 2 / (2 * Xstd ** 2))
            gwX = np.product(gX, 1)
            gw[point, :] = gwX / gwX.sum()
            nearest[point, :] = ind
        gw[np.isnan(gw)] = 0.01
    return gw, nearest.astype(int)

def planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array, ww=0): # assuming only cupy arrays, if GPU
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py

    try:
        inclination = inclination * jnp.pi / 180.0
        periastron = periastron * jnp.pi / 180.0
        ww = ww * jnp.pi / 180.0

        case_circular = (eccentricity == 0) * (ww == 0)
        case_not_circular = (eccentricity != 0) + (ww != 0)

        aa = jnp.where(periastron < np.pi / 2, 1.0 * jnp.pi / 2 - periastron, 5.0 * jnp.pi / 2 - periastron)
        bb = jnp.where(case_circular, sma_over_rs * jnp.cos(2 * jnp.pi * (time_array - mid_time) / period), 2 * jnp.arctan(jnp.sqrt((1 - eccentricity) / (1 + eccentricity)) * jnp.tan(aa / 2)))

        bb = jnp.where(case_not_circular * (bb < 0), bb + 2 * jnp.pi, bb)

        mid_time = mid_time.astype(jnp.float64) - (period / 2.0 / jnp.pi) * (bb - eccentricity * jnp.sin(bb))
        m = (time_array - mid_time - jnp.int_((time_array - mid_time) / period) * period) * 2.0 * jnp.pi / period
        u0 = m
        u1 = 0
        def cond(u0, u1, ii):
            return (jnp.abs(u1 - u0) > 10 ** (-6)).all()
        def body(u0, u1, ii):
            u1 = u0 - (u0 - eccentricity * jnp.sin(u0) - m) / (1 - eccentricity * jnp.cos(u0))
            u0 = u1
            ii += 1
            if (ii == 10000): # setting a limit of 1k iterations - arbitrary limit
                raise RuntimeError('Failed to find a solution in 10000 loops')
            return u1
        u1 = jax.lax.while_loop(cond_fun=cond, body_fun=body, init_val=(u0, u1, 0))
        
        vv = jnp.where(case_circular, 2 * jnp.pi * (time_array - mid_time) / period, 2 * jnp.arctan(jnp.sqrt((1 + eccentricity) / (1 - eccentricity)) * jnp.tan((u1) / 2)))
        
        rr = jnp.where(case_circular, 0., sma_over_rs * (1 - (eccentricity ** 2)) / (jnp.ones_like(vv) + eccentricity * jnp.cos(vv)))

        aa = jnp.where(case_circular, 0., jnp.cos(vv + periastron))
        bb = jnp.where(case_circular, bb, jnp.sin(vv + periastron))

        x = jnp.where(case_circular, bb * jnp.sin(inclination), rr * bb * jnp.sin(inclination))
        y = jnp.where(case_circular, sma_over_rs * jnp.sin(vv), rr * (-aa * jnp.cos(ww) + bb * jnp.sin(ww) * jnp.cos(inclination)))
        z = jnp.where(case_circular, - bb * jnp.cos(inclination), rr * (-aa * jnp.sin(ww) - bb * jnp.cos(ww) * jnp.cos(inclination)))
    except NameError:
        inclination = inclination * np.pi / 180.0
        periastron = periastron * np.pi / 180.0
        ww = ww * np.pi / 180.0

        if eccentricity == 0 and ww == 0:
            vv = 2 * np.pi * (time_array - mid_time) / period
            bb = sma_over_rs * np.cos(vv)
            return [bb * np.sin(inclination), sma_over_rs * np.sin(vv), - bb * np.cos(inclination)]

        if periastron < np.pi / 2:
            aa = 1.0 * np.pi / 2 - periastron
        else:
            aa = 5.0 * np.pi / 2 - periastron
        bb = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(aa / 2))
        if bb < 0:
            bb += 2 * np.pi
        mid_time = float(mid_time) - (period / 2.0 / np.pi) * (bb - eccentricity * np.sin(bb))
        m = (time_array - mid_time - np.int_((time_array - mid_time) / period) * period) * 2.0 * np.pi / period
        u0 = m
        stop = False
        u1 = 0
        for ii in range(10000):  # setting a limit of 1k iterations - arbitrary limit
            u1 = u0 - (u0 - eccentricity * np.sin(u0) - m) / (1 - eccentricity * np.cos(u0))
            stop = (np.abs(u1 - u0) < 10 ** (-6)).all()
            if stop:
                break
            else:
                u0 = u1
        if not stop:
            raise RuntimeError('Failed to find a solution in 10000 loops')

        vv = 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan((u1) / 2))
        rr = sma_over_rs * (1 - (eccentricity ** 2)) / (np.ones_like(vv) + eccentricity * np.cos(vv))

        aa = np.cos(vv + periastron)
        bb = np.sin(vv + periastron)

        x = rr * bb * np.sin(inclination)
        y = rr * (-aa * np.cos(ww) + bb * np.sin(ww) * np.cos(inclination))
        z = rr * (-aa * np.sin(ww) - bb * np.cos(ww) * np.cos(inclination))

    return [x, y, z]

#@maybe_decorate(jit(parallel=True, cache=True))
#@maybe_decorate(lambda x: x)
def integral_r_claret(limb_darkening_coefficients, r):
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py
    a1, a2, a3, a4 = limb_darkening_coefficients
    mu44 = 1.0 - r * r
    try:
        mu24 = jnp.sqrt(mu44)
        mu14 = jnp.sqrt(mu24)
    except NameError:
        mu24 = np.sqrt(mu44)
        mu14 = np.sqrt(mu24)
    return - (2.0 * (1.0 - a1 - a2 - a3 - a4) / 4) * mu44 \
           - (2.0 * a1 / 5) * mu44 * mu14 \
           - (2.0 * a2 / 6) * mu44 * mu24 \
           - (2.0 * a3 / 7) * mu44 * mu24 * mu14 \
           - (2.0 * a4 / 8) * mu44 * mu44
    
integral_r = {
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py
    'claret': integral_r_claret,
    #'linear': integral_r_linear,
    #'quad': integral_r_quad,
    #'sqrt': integral_r_sqrt,
    #'zero': integral_r_zero
}

# coefficients from https://pomax.github.io/bezierinfo/legendre-gauss.html
# please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/analysis/numerical_integration.py
gauss0 = [
    [1.0000000000000000, -0.5773502691896257],
    [1.0000000000000000, 0.5773502691896257]
]

gauss10 = [
    [0.2955242247147529, -0.1488743389816312],
    [0.2955242247147529, 0.1488743389816312],
    [0.2692667193099963, -0.4333953941292472],
    [0.2692667193099963, 0.4333953941292472],
    [0.2190863625159820, -0.6794095682990244],
    [0.2190863625159820, 0.6794095682990244],
    [0.1494513491505806, -0.8650633666889845],
    [0.1494513491505806, 0.8650633666889845],
    [0.0666713443086881, -0.9739065285171717],
    [0.0666713443086881, 0.9739065285171717]
]

gauss20 = [
    [0.1527533871307258, -0.0765265211334973],
    [0.1527533871307258, 0.0765265211334973],
    [0.1491729864726037, -0.2277858511416451],
    [0.1491729864726037, 0.2277858511416451],
    [0.1420961093183820, -0.3737060887154195],
    [0.1420961093183820, 0.3737060887154195],
    [0.1316886384491766, -0.5108670019508271],
    [0.1316886384491766, 0.5108670019508271],
    [0.1181945319615184, -0.6360536807265150],
    [0.1181945319615184, 0.6360536807265150],
    [0.1019301198172404, -0.7463319064601508],
    [0.1019301198172404, 0.7463319064601508],
    [0.0832767415767048, -0.8391169718222188],
    [0.0832767415767048, 0.8391169718222188],
    [0.0626720483341091, -0.9122344282513259],
    [0.0626720483341091, 0.9122344282513259],
    [0.0406014298003869, -0.9639719272779138],
    [0.0406014298003869, 0.9639719272779138],
    [0.0176140071391521, -0.9931285991850949],
    [0.0176140071391521, 0.9931285991850949],
]

gauss30 = [
    [0.1028526528935588, -0.0514718425553177],
    [0.1028526528935588, 0.0514718425553177],
    [0.1017623897484055, -0.1538699136085835],
    [0.1017623897484055, 0.1538699136085835],
    [0.0995934205867953, -0.2546369261678899],
    [0.0995934205867953, 0.2546369261678899],
    [0.0963687371746443, -0.3527047255308781],
    [0.0963687371746443, 0.3527047255308781],
    [0.0921225222377861, -0.4470337695380892],
    [0.0921225222377861, 0.4470337695380892],
    [0.0868997872010830, -0.5366241481420199],
    [0.0868997872010830, 0.5366241481420199],
    [0.0807558952294202, -0.6205261829892429],
    [0.0807558952294202, 0.6205261829892429],
    [0.0737559747377052, -0.6978504947933158],
    [0.0737559747377052, 0.6978504947933158],
    [0.0659742298821805, -0.7677774321048262],
    [0.0659742298821805, 0.7677774321048262],
    [0.0574931562176191, -0.8295657623827684],
    [0.0574931562176191, 0.8295657623827684],
    [0.0484026728305941, -0.8825605357920527],
    [0.0484026728305941, 0.8825605357920527],
    [0.0387991925696271, -0.9262000474292743],
    [0.0387991925696271, 0.9262000474292743],
    [0.0287847078833234, -0.9600218649683075],
    [0.0287847078833234, 0.9600218649683075],
    [0.0184664683110910, -0.9836681232797472],
    [0.0184664683110910, 0.9836681232797472],
    [0.0079681924961666, -0.9968934840746495],
    [0.0079681924961666, 0.9968934840746495]
]

gauss40 = [
    [0.0775059479784248, -0.0387724175060508],
    [0.0775059479784248, 0.0387724175060508],
    [0.0770398181642480, -0.1160840706752552],
    [0.0770398181642480, 0.1160840706752552],
    [0.0761103619006262, -0.1926975807013711],
    [0.0761103619006262, 0.1926975807013711],
    [0.0747231690579683, -0.2681521850072537],
    [0.0747231690579683, 0.2681521850072537],
    [0.0728865823958041, -0.3419940908257585],
    [0.0728865823958041, 0.3419940908257585],
    [0.0706116473912868, -0.4137792043716050],
    [0.0706116473912868, 0.4137792043716050],
    [0.0679120458152339, -0.4830758016861787],
    [0.0679120458152339, 0.4830758016861787],
    [0.0648040134566010, -0.5494671250951282],
    [0.0648040134566010, 0.5494671250951282],
    [0.0613062424929289, -0.6125538896679802],
    [0.0613062424929289, 0.6125538896679802],
    [0.0574397690993916, -0.6719566846141796],
    [0.0574397690993916, 0.6719566846141796],
    [0.0532278469839368, -0.7273182551899271],
    [0.0532278469839368, 0.7273182551899271],
    [0.0486958076350722, -0.7783056514265194],
    [0.0486958076350722, 0.7783056514265194],
    [0.0438709081856733, -0.8246122308333117],
    [0.0438709081856733, 0.8246122308333117],
    [0.0387821679744720, -0.8659595032122595],
    [0.0387821679744720, 0.8659595032122595],
    [0.0334601952825478, -0.9020988069688743],
    [0.0334601952825478, 0.9020988069688743],
    [0.0279370069800234, -0.9328128082786765],
    [0.0279370069800234, 0.9328128082786765],
    [0.0222458491941670, -0.9579168192137917],
    [0.0222458491941670, 0.9579168192137917],
    [0.0164210583819079, -0.9772599499837743],
    [0.0164210583819079, 0.9772599499837743],
    [0.0104982845311528, -0.9907262386994570],
    [0.0104982845311528, 0.9907262386994570],
    [0.0045212770985332, -0.9982377097105593],
    [0.0045212770985332, 0.9982377097105593],
]

gauss50 = [
    [0.0621766166553473, -0.0310983383271889],
    [0.0621766166553473, 0.0310983383271889],
    [0.0619360674206832, -0.0931747015600861],
    [0.0619360674206832, 0.0931747015600861],
    [0.0614558995903167, -0.1548905899981459],
    [0.0614558995903167, 0.1548905899981459],
    [0.0607379708417702, -0.2160072368760418],
    [0.0607379708417702, 0.2160072368760418],
    [0.0597850587042655, -0.2762881937795320],
    [0.0597850587042655, 0.2762881937795320],
    [0.0586008498132224, -0.3355002454194373],
    [0.0586008498132224, 0.3355002454194373],
    [0.0571899256477284, -0.3934143118975651],
    [0.0571899256477284, 0.3934143118975651],
    [0.0555577448062125, -0.4498063349740388],
    [0.0555577448062125, 0.4498063349740388],
    [0.0537106218889962, -0.5044581449074642],
    [0.0537106218889962, 0.5044581449074642],
    [0.0516557030695811, -0.5571583045146501],
    [0.0516557030695811, 0.5571583045146501],
    [0.0494009384494663, -0.6077029271849502],
    [0.0494009384494663, 0.6077029271849502],
    [0.0469550513039484, -0.6558964656854394],
    [0.0469550513039484, 0.6558964656854394],
    [0.0443275043388033, -0.7015524687068222],
    [0.0443275043388033, 0.7015524687068222],
    [0.0415284630901477, -0.7444943022260685],
    [0.0415284630901477, 0.7444943022260685],
    [0.0385687566125877, -0.7845558329003993],
    [0.0385687566125877, 0.7845558329003993],
    [0.0354598356151462, -0.8215820708593360],
    [0.0354598356151462, 0.8215820708593360],
    [0.0322137282235780, -0.8554297694299461],
    [0.0322137282235780, 0.8554297694299461],
    [0.0288429935805352, -0.8859679795236131],
    [0.0288429935805352, 0.8859679795236131],
    [0.0253606735700124, -0.9130785566557919],
    [0.0253606735700124, 0.9130785566557919],
    [0.0217802431701248, -0.9366566189448780],
    [0.0217802431701248, 0.9366566189448780],
    [0.0181155607134894, -0.9566109552428079],
    [0.0181155607134894, 0.9566109552428079],
    [0.0143808227614856, -0.9728643851066920],
    [0.0143808227614856, 0.9728643851066920],
    [0.0105905483836510, -0.9853540840480058],
    [0.0105905483836510, 0.9853540840480058],
    [0.0067597991957454, -0.9940319694320907],
    [0.0067597991957454, 0.9940319694320907],
    [0.0029086225531551, -0.9988664044200710],
    [0.0029086225531551, 0.9988664044200710]
]

gauss60 = [
    [0.0519078776312206, -0.0259597723012478],
    [0.0519078776312206, 0.0259597723012478],
    [0.0517679431749102, -0.0778093339495366],
    [0.0517679431749102, 0.0778093339495366],
    [0.0514884515009809, -0.1294491353969450],
    [0.0514884515009809, 0.1294491353969450],
    [0.0510701560698556, -0.1807399648734254],
    [0.0510701560698556, 0.1807399648734254],
    [0.0505141845325094, -0.2315435513760293],
    [0.0505141845325094, 0.2315435513760293],
    [0.0498220356905502, -0.2817229374232617],
    [0.0498220356905502, 0.2817229374232617],
    [0.0489955754557568, -0.3311428482684482],
    [0.0489955754557568, 0.3311428482684482],
    [0.0480370318199712, -0.3796700565767980],
    [0.0480370318199712, 0.3796700565767980],
    [0.0469489888489122, -0.4271737415830784],
    [0.0469489888489122, 0.4271737415830784],
    [0.0457343797161145, -0.4735258417617071],
    [0.0457343797161145, 0.4735258417617071],
    [0.0443964787957871, -0.5186014000585697],
    [0.0443964787957871, 0.5186014000585697],
    [0.0429388928359356, -0.5622789007539445],
    [0.0429388928359356, 0.5622789007539445],
    [0.0413655512355848, -0.6044405970485104],
    [0.0413655512355848, 0.6044405970485104],
    [0.0396806954523808, -0.6449728284894770],
    [0.0396806954523808, 0.6449728284894770],
    [0.0378888675692434, -0.6837663273813555],
    [0.0378888675692434, 0.6837663273813555],
    [0.0359948980510845, -0.7207165133557304],
    [0.0359948980510845, 0.7207165133557304],
    [0.0340038927249464, -0.7557237753065856],
    [0.0340038927249464, 0.7557237753065856],
    [0.0319212190192963, -0.7886937399322641],
    [0.0319212190192963, 0.7886937399322641],
    [0.0297524915007889, -0.8195375261621458],
    [0.0297524915007889, 0.8195375261621458],
    [0.0275035567499248, -0.8481719847859296],
    [0.0275035567499248, 0.8481719847859296],
    [0.0251804776215212, -0.8745199226468983],
    [0.0251804776215212, 0.8745199226468983],
    [0.0227895169439978, -0.8985103108100460],
    [0.0227895169439978, 0.8985103108100460],
    [0.0203371207294573, -0.9200784761776275],
    [0.0203371207294573, 0.9200784761776275],
    [0.0178299010142077, -0.9391662761164232],
    [0.0178299010142077, 0.9391662761164232],
    [0.0152746185967848, -0.9557222558399961],
    [0.0152746185967848, 0.9557222558399961],
    [0.0126781664768160, -0.9697017887650528],
    [0.0126781664768160, 0.9697017887650528],
    [0.0100475571822880, -0.9810672017525982],
    [0.0100475571822880, 0.9810672017525982],
    [0.0073899311633455, -0.9897878952222218],
    [0.0073899311633455, 0.9897878952222218],
    [0.0047127299269536, -0.9958405251188381],
    [0.0047127299269536, 0.9958405251188381],
    [0.0020268119688738, -0.9992101232274361],
    [0.0020268119688738, 0.9992101232274361],
]

# please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/analysis/numerical_integration.py
try:
    gauss_table = [jnp.swapaxes(jnp.array(gauss0, dtype=jnp.float64), 0, 1), jnp.swapaxes(jnp.array(gauss10, dtype=jnp.float64), 0, 1), 
               jnp.swapaxes(jnp.array(gauss20, dtype=jnp.float64), 0, 1), jnp.swapaxes(jnp.array(gauss30, dtype=jnp.float64), 0, 1), 
               jnp.swapaxes(jnp.array(gauss40, dtype=jnp.float64), 0, 1), jnp.swapaxes(jnp.array(gauss50, dtype=jnp.float64), 0, 1),
               jnp.swapaxes(jnp.array(gauss60, dtype=jnp.float64), 0, 1)]
except NameError:
    gauss_table = [np.swapaxes(np.array(gauss0, dtype=np.float64), 0, 1), np.swapaxes(np.array(gauss10, dtype=np.float64), 0, 1), 
               np.swapaxes(np.array(gauss20, dtype=np.float64), 0, 1), np.swapaxes(np.array(gauss30, dtype=np.float64), 0, 1), 
               np.swapaxes(np.array(gauss40, dtype=np.float64), 0, 1), np.swapaxes(np.array(gauss50, dtype=np.float64), 0, 1),
               np.swapaxes(np.array(gauss60, dtype=np.float64), 0, 1)]

#@jax.jit
def gauss_numerical_integration(
    #f, 
    x1, x2, precision, *f_args):
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/analysis/numerical_integration.py
    x1, x2 = (x2 - x1) / 2, (x2 + x1) / 2

    try:
        return x1 * jnp.sum(gauss_table[precision][0][:, None] *
                       #f(x1[None, :] * gauss_table[precision][1][:, None] + x2[None, :], *f_args), 0)
                       num_claret(x1[None, :] * gauss_table[precision][1][:, None] + x2[None, :], *f_args), 0)
    except NameError:
        return x1 * np.sum(gauss_table[precision][0][:, None] *
                       #f(x1[None, :] * gauss_table[precision][1][:, None] + x2[None, :], *f_args), 0)
                       num_claret(x1[None, :] * gauss_table[precision][1][:, None] + x2[None, :], *f_args), 0)

#@maybe_decorate(lambda x: x)
def num_claret(r, limb_darkening_coefficients, rprs, z):
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py
    a1, a2, a3, a4 = limb_darkening_coefficients
    rsq = r * r
    mu44 = 1.0 - rsq
    try:
        mu24 = jnp.sqrt(mu44)
        mu14 = jnp.sqrt(mu24)
        return ((1.0 - a1 - a2 - a3 - a4) + a1 * mu14 + a2 * mu24 + a3 * mu24 * mu14 + a4 * mu44) \
            * r * jnp.arccos(jnp.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))
    except NameError:
        mu24 = np.sqrt(mu44)
        mu14 = np.sqrt(mu24)
        return ((1.0 - a1 - a2 - a3 - a4) + a1 * mu14 + a2 * mu24 + a3 * mu24 * mu14 + a4 * mu44) \
            * r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))

#@jax.jit
def integral_r_f_claret(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py
    return gauss_numerical_integration(
        #num_claret, 
        r1, r2, precision, limb_darkening_coefficients, rprs, z)

integral_r_f = {
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py
    'claret': integral_r_f_claret,
    #'linear': integral_r_f_linear,
    #'quad': integral_r_f_quad,
    #'sqrt': integral_r_f_sqrt,
    #'zero': integral_r_f_zero,
}

#@jax.jit
def integral_centred(
    #method, 
    limb_darkening_coefficients, rprs, ww1, ww2): # assuming only cupy arrays, if GPU
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py
    method='claret'

    try:
        return jnp.abs(ww2 - ww1) * (integral_r[method](limb_darkening_coefficients, rprs)
            - integral_r[method](limb_darkening_coefficients, 0.0))
    except NameError:
        return (integral_r[method](limb_darkening_coefficients, rprs)
            - integral_r[method](limb_darkening_coefficients, 0.0)) * np.abs(ww2 - ww1)

#@jax.jit
def integral_plus_core(
    #method, 
    limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3): # assuming only cupy arrays, if GPU
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py
    method='claret'

    if len(z) == 0:
        return z
    try:
        rr1 = z * jnp.cos(ww1) + jnp.sqrt(jnp.maximum(rprs ** 2 - (z * jnp.sin(ww1)) ** 2, 0))
        rr1 = jnp.clip(rr1, 0, 1)
        rr2 = z * jnp.cos(ww2) + jnp.sqrt(jnp.maximum(rprs ** 2 - (z * jnp.sin(ww2)) ** 2, 0))
        rr2 = jnp.clip(rr2, 0, 1)
        w1 = jnp.minimum(ww1, ww2)
        r1 = jnp.minimum(rr1, rr2)
        w2 = jnp.maximum(ww1, ww2)
        r2 = jnp.maximum(rr1, rr2)
    except NameError:
        rr1 = z * np.cos(ww1) + np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww1)) ** 2, 0))
        rr1 = np.clip(rr1, 0, 1)
        rr2 = z * np.cos(ww2) + np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww2)) ** 2, 0))
        rr2 = np.clip(rr2, 0, 1)
        w1 = np.minimum(ww1, ww2)
        r1 = np.minimum(rr1, rr2)
        w2 = np.maximum(ww1, ww2)
        r2 = np.maximum(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, 0.0) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * w2
    partc = integral_r[method](limb_darkening_coefficients, r2) * (-w1)
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc + partd

#@jax.jit
def integral_minus_core(
    #method, 
    limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3): # assuming only cupy arrays, if GPU
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py
    method='claret'

    if len(z) == 0:
        return z
    try:
        rr1 = z * jnp.cos(ww1) - jnp.sqrt(jnp.maximum(rprs ** 2 - (z * jnp.sin(ww1)) ** 2, 0))
        rr1 = jnp.clip(rr1, 0, 1)
        rr2 = z * jnp.cos(ww2) - jnp.sqrt(jnp.maximum(rprs ** 2 - (z * jnp.sin(ww2)) ** 2, 0))
        rr2 = jnp.clip(rr2, 0, 1)
        w1 = jnp.minimum(ww1, ww2)
        r1 = jnp.minimum(rr1, rr2)
        w2 = jnp.maximum(ww1, ww2)
        r2 = jnp.maximum(rr1, rr2)
    except NameError:
        rr1 = z * np.cos(ww1) - np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww1)) ** 2, 0))
        rr1 = np.clip(rr1, 0, 1)
        rr2 = z * np.cos(ww2) - np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww2)) ** 2, 0))
        rr2 = np.clip(rr2, 0, 1)
        w1 = np.minimum(ww1, ww2)
        r1 = np.minimum(rr1, rr2)
        w2 = np.maximum(ww1, ww2)
        r2 = np.maximum(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, 0.0) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * (-w1)
    partc = integral_r[method](limb_darkening_coefficients, r2) * w2
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc - partd

def transit_flux_drop(limb_darkening_coefficients, rp_over_rs, z_over_rs, 
                      #method='claret', 
                      precision=3): # assuming only cupy arrays, if GPU
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py

    try:
        z_over_rs = jnp.where(z_over_rs < 0, 1.0 + 100.0 * rp_over_rs, z_over_rs)
        z_over_rs = jnp.maximum(z_over_rs, 10**(-10))
    except NameError:
        z_over_rs = np.where(z_over_rs < 0, 1.0 + 100.0 * rp_over_rs, z_over_rs)
        z_over_rs = np.maximum(z_over_rs, 10**(-10))

    # cases
    zsq = z_over_rs * z_over_rs
    sum_z_rprs = z_over_rs + rp_over_rs
    dif_z_rprs = rp_over_rs - z_over_rs
    sqr_dif_z_rprs = zsq - rp_over_rs ** 2
    try:
        case0 = jnp.where((z_over_rs == 0) & (rp_over_rs <= 1))
        case1 = jnp.where((z_over_rs < rp_over_rs) & (sum_z_rprs <= 1))
        casea = jnp.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs < 1))
        caseb = jnp.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs > 1))
        case2 = jnp.where((z_over_rs == rp_over_rs) & (sum_z_rprs <= 1))
        casec = jnp.where((z_over_rs == rp_over_rs) & (sum_z_rprs > 1))
        case3 = jnp.where((z_over_rs > rp_over_rs) & (sum_z_rprs < 1))
        case4 = jnp.where((z_over_rs > rp_over_rs) & (sum_z_rprs == 1))
        case5 = jnp.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs < 1))
        case6 = jnp.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs == 1))
        case7 = jnp.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs > 1) & (-1 < dif_z_rprs))
        plus_case = jnp.concatenate((case1[0], case2[0], case3[0], case4[0], case5[0], casea[0], casec[0]))
        minus_case = jnp.concatenate((case3[0], case4[0], case5[0], case6[0], case7[0]))
        star_case = jnp.concatenate((case5[0], case6[0], case7[0], casea[0], casec[0]))
    except NameError:
        case0 = np.where((z_over_rs == 0) & (rp_over_rs <= 1))
        case1 = np.where((z_over_rs < rp_over_rs) & (sum_z_rprs <= 1))
        casea = np.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs < 1))
        caseb = np.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs > 1))
        case2 = np.where((z_over_rs == rp_over_rs) & (sum_z_rprs <= 1))
        casec = np.where((z_over_rs == rp_over_rs) & (sum_z_rprs > 1))
        case3 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs < 1))
        case4 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs == 1))
        case5 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs < 1))
        case6 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs == 1))
        case7 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs > 1) & (-1 < dif_z_rprs))
        plus_case = np.concatenate((case1[0], case2[0], case3[0], case4[0], case5[0], casea[0], casec[0]))
        minus_case = np.concatenate((case3[0], case4[0], case5[0], case6[0], case7[0]))
        star_case = np.concatenate((case5[0], case6[0], case7[0], casea[0], casec[0]))

    # cross points
    try:
        ph = jnp.arccos(jnp.clip((1.0 - rp_over_rs ** 2 + zsq) / (2.0 * z_over_rs), -1, 1))
        theta_1 = jnp.zeros(len(z_over_rs))
        ph_case = jnp.concatenate((case5[0], casea[0], casec[0]))
        theta_1.at[ph_case].set(ph[ph_case])
        theta_2 = jnp.arcsin(jnp.minimum(rp_over_rs / z_over_rs, 1))
        theta_2.at[case1].set(jnp.pi)
        theta_2.at[case2].set(jnp.pi / 2.0)
        theta_2.at[casea].set(jnp.pi)
        theta_2.at[casec].set(jnp.pi / 2.0)
        theta_2.at[case7].set(ph[case7])
    except NameError:
        ph = np.arccos(np.clip((1.0 - rp_over_rs ** 2 + zsq) / (2.0 * z_over_rs), -1, 1))
        theta_1 = np.zeros(len(z_over_rs))
        ph_case = np.concatenate((case5[0], casea[0], casec[0]))
        theta_1[ph_case] = ph[ph_case]
        theta_2 = np.arcsin(np.minimum(rp_over_rs / z_over_rs, 1))
        theta_2[case1] = np.pi
        theta_2[case2] = np.pi / 2.0
        theta_2[casea] = np.pi
        theta_2[casec] = np.pi / 2.0
        theta_2[case7] = ph[case7]

    # flux_upper
    try:
        plusflux = jnp.zeros(len(z_over_rs))
        plusflux.at[plus_case].set(integral_plus_core(
            #method, 
            limb_darkening_coefficients, rp_over_rs, z_over_rs[plus_case],
            theta_1[plus_case], theta_2[plus_case], precision=precision))
        if len(case0[0]) > 0:
            plusflux.at[case0].set(integral_centred(
                #method, 
                limb_darkening_coefficients, rp_over_rs, 0.0, jnp.pi))
        if len(caseb[0]) > 0:
            plusflux.at[caseb].set(integral_centred(
                #method, 
                limb_darkening_coefficients, 1, 0.0, jnp.pi))
    except NameError:
        plusflux = np.zeros(len(z_over_rs))
        plusflux[plus_case] = integral_plus_core(
            #method, 
            limb_darkening_coefficients, rp_over_rs, z_over_rs[plus_case],
            theta_1[plus_case], theta_2[plus_case], precision=precision)
        if len(case0[0]) > 0:
            plusflux[case0] = integral_centred(
                #method, 
                limb_darkening_coefficients, rp_over_rs, 0.0, np.pi)
        if len(caseb[0]) > 0:
            plusflux[caseb] = integral_centred(
                #method, 
                limb_darkening_coefficients, 1, 0.0, np.pi)

    # flux_lower
        
    try:
        minsflux = jnp.zeros(len(z_over_rs))
        minsflux.at[minus_case].set(integral_minus_core(
            #method, 
            limb_darkening_coefficients, rp_over_rs,
            z_over_rs[minus_case], 0.0, theta_2[minus_case], precision=precision))
    except NameError:
        minsflux = np.zeros(len(z_over_rs))
        minsflux[minus_case] = integral_minus_core(
            #method, 
            limb_darkening_coefficients, rp_over_rs,
            z_over_rs[minus_case], 0.0, theta_2[minus_case], precision=precision)

    # flux_star
    try:
        starflux = jnp.zeros(len(z_over_rs))
        starflux.at[star_case].set(integral_centred(
            #method, 
            limb_darkening_coefficients, 1, 0.0, ph[star_case]))
    except NameError:
        starflux = np.zeros(len(z_over_rs))
        starflux[star_case] = integral_centred(
            #method, 
            limb_darkening_coefficients, 1, 0.0, ph[star_case])

    # flux_total
    try:
        total_flux = integral_centred(
            #method, 
            limb_darkening_coefficients, 1, 0.0, 2.0 * jnp.pi)
    except NameError:
        total_flux = integral_centred(
            #method, 
            limb_darkening_coefficients, 1, 0.0, 2.0 * np.pi)

    return 1 - (2.0 / total_flux) * (plusflux + starflux - minsflux)

def pytransit(limb_darkening_coefficients, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
            mid_time, time_array, 
            #method='claret', 
            precision=3): # assuming only cupy arrays, if GPU
    # please see original: https://github.com/ucl-exoplanets/pylightcurve/blob/master/pylightcurve/models/exoplanet_lc.py

    position_vector = planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array)

    try:
        projected_distance = jnp.where(
            position_vector[0] < 0, 1.0 + 5.0 * rp_over_rs,
            jnp.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))
    except NameError:
        projected_distance = np.where(
            position_vector[0] < 0, 1.0 + 5.0 * rp_over_rs,
            np.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))

    return transit_flux_drop(limb_darkening_coefficients, rp_over_rs, projected_distance,
                             #method=method, 
                             precision=precision)

def transit(times, values): # assuming only cupy arrays, if GPU
    try:
        #print(torch.from_dlpack(np.array([values['u0'], values['u1'], values['u2'], values['u3']], dtype=np.float64)))
        #print(torch.from_dlpack(np.array(values['rprs'], dtype=np.float64)))
        #print(torch.from_dlpack(np.array(values['per'], dtype=np.float64)))
        #print(torch.from_dlpack(np.array(values['ars'], dtype=np.float64)))
        #print(torch.from_dlpack(np.array(values['ecc'], dtype=np.float64)))
        #print(torch.from_dlpack(np.array(values['inc'], dtype=np.float64)))
        #print(torch.from_dlpack(np.array(values['omega'], dtype=np.float64)))
        #print(torch.from_dlpack(np.array(values['tmid'], dtype=np.float64)))
        #print(torch.from_dlpack(np.array(times, dtype=np.float64)))
        #print(np.array(values['rprs'], dtype=np.float64).size)
        #model = pytransit('claret', torch.from_dlpack(np.array([values['u0'], values['u1'], values['u2'], values['u3']], dtype=np.float64)), 
        #              torch.from_dlpack(np.array(values['rprs'], dtype=np.float64)), torch.from_dlpack(np.array(values['per'], dtype=np.float64)), 
        #              torch.from_dlpack(np.array(values['ars'], dtype=np.float64)), torch.from_dlpack(np.array(values['ecc'], dtype=np.float64)), 
        #              torch.from_dlpack(np.array(values['inc'], dtype=np.float64)), torch.from_dlpack(np.array(values['omega'], dtype=np.float64)),
        #              torch.from_dlpack(np.array(values['tmid'], dtype=np.float64)), torch.from_dlpack(np.array(times, dtype=np.float64)), 
        #              precision=3, n_pars=np.array(values['rprs'], dtype=np.float64).size) #pylightcurve-torch has a different syntax, and requires PyTorch Tensors instead of Nympy arrays
        #return np.asnumpy(np.from_dlpack(model)) # must convert back from PyTorch GPU Tensors to Numpy arrays for CPU
        #model = pytransit(np.asnumpy(np.array([values['u0'], values['u1'], values['u2'], values['u3']], dtype=np.float64)),
        #              np.asnumpy(values['rprs']), np.asnumpy(values['per']), np.asnumpy(values['ars']),
        #              np.asnumpy(values['ecc']), np.asnumpy(values['inc']), np.asnumpy(values['omega']),
        #              np.asnumpy(values['tmid']), np.asnumpy(times), method='claret', precision=3)
        #return np.array(model, dtype=np.float64) # must convert back from Numpy array to cupy array for GPU
        #jax.device_put(times)
        #for k in values.keys():
        #    jax.device_put(values[k])
        mmodel = pytransit([values['u0'], values['u1'], values['u2'], values['u3']],
                      values['rprs'], values['per'], values['ars'],
                      values['ecc'], values['inc'], values['omega'],
                      values['tmid'], times, 
                      #method='claret', 
                      precision=3)
        return np.from_dlpack(model.toDlpack()) # must convert back from JAX tracer to Numpy array for CPU
    #except AttributeError:
    #except TypeError:
    except NameError:
        model = pytransit([values['u0'], values['u1'], values['u2'], values['u3']],
                      values['rprs'], values['per'], values['ars'],
                      values['ecc'], values['inc'], values['omega'],
                      values['tmid'], times, 
                      #method='claret', 
                      precision=3)
        return model

@jit(nopython=True)
def get_phase(times, per, tmid):
    return (times - tmid + 0.25 * per) / per % 1 - 0.25

def mc_a1(m_a2, sig_a2, transit, airmass, data, n=10000): # assuming only cupy arrays, if GPU
    a2 = np.random.normal(m_a2, sig_a2, n)
    model = transit * np.exp(np.repeat(np.expand_dims(a2, 0), airmass.shape[0], 0).T * airmass)
    detrend = data / model
    return np.mean(np.median(detrend, 0)), np.std(np.median(detrend, 0))

def round_to_2(*args):
    x = args[0]
    if len(args) == 1:
        y = args[0]
    else:
        y = args[1]
    if np.floor(y) >= 1.:
        roundval = 2
    else:
        try:
            roundval = -int(np.floor(np.log10(abs(y)))) + 1
        except:
            roundval = 1
    return round(x, roundval)

# average data into bins of dt from start to finish
def time_bin(time, flux, dt=1. / (60 * 24)): # assuming only cupy arrays, if GPU
    bins = int(np.floor((max(time) - min(time)) / dt))
    bflux = np.zeros(bins)
    btime = np.zeros(bins)
    bstds = np.zeros(bins)
    for i in range(bins):
        mask = (time >= (min(time) + i * dt)) & (time < (min(time) + (i + 1) * dt))
        if mask.sum() > 0:
            bflux[i] = np.nanmean(flux[mask])
            btime[i] = np.nanmean(time[mask])
            bstds[i] = np.nanstd(flux[mask]) / (mask.sum() ** 0.5)
    zmask = (bflux == 0) | (btime == 0) | np.isnan(bflux) | np.isnan(btime)
    return btime[~zmask], bflux[~zmask], bstds[~zmask]

@jit(nopython=True)
# Function that bins an array
def binner(arr, n, err=''): # assuming only cupy arrays, if GPU
    if len(err) == 0:
        ecks = np.pad(arr.astype(float), (0, ((n - arr.size % n) % n)), mode='constant',
                      constant_values=np.NaN).reshape(-1, n)
        arr = np.nanmean(ecks, axis=1)
        return arr
    else:
        ecks = np.pad(arr.astype(float), (0, ((n - arr.size % n) % n)), mode='constant',
                      constant_values=np.NaN).reshape(-1, n)
        why = np.pad(err.astype(float), (0, ((n - err.size % n) % n)), mode='constant', constant_values=np.NaN).reshape(
            -1, n)
        weights = 1. / (why ** 2.)
        # Calculate the weighted average
        arr = np.nansum(ecks * weights, axis=1) / np.nansum(weights, axis=1)
        err = np.array([np.sqrt(1. / np.nansum(1. / (np.array(i) ** 2.))) for i in why])
        return arr, err


class lc_fitter(object):

    def __init__(self, time, data, dataerr, airmass, prior, bounds, neighbors=200, mode='ns', verbose=True):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.airmass = airmass
        self.prior = prior
        self.bounds = bounds
        self.max_ncalls = 2e5
        self.verbose = verbose
        self.mode = mode
        self.neighbors = neighbors
        self.results = None
        if self.mode == "lm":
            self.fit_LM()
        elif self.mode == "ns":
            self.fit_nested()

    def fit_LM(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])

        # trim data around predicted transit/eclipse time
        if np.ndim(self.airmass) == 2:
            print(f'Computing nearest neighbors and gaussian weights for {len(self.time)} npts...')
            try:
                self.gw, self.nearest = np.asnumpy(gaussian_weights(np.array(self.airmass), neighbors=np.array(self.neighbors)))
            except AttributeError:
                self.gw, self.nearest = gaussian_weights(self.airmass, neighbors=self.neighbors)

        def lc2min_nneighbor(pars):
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]
            lightcurve = transit(self.time, self.prior)
            detrended = self.data / lightcurve
            try:
                wf = np.asnumpy(weightedflux(np.array(detrended), np.array(self.gw), np.array(self.nearest)))
            except AttributeError:
                wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve * wf
            return ((self.data - model) / self.dataerr) ** 2

        def lc2min_airmass(pars):
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]
            model = transit(self.time, self.prior)
            model *= self.prior['a1'] * np.exp(self.prior['a2'] * self.airmass)
            return ((self.data - model) / self.dataerr) ** 2

        try:
            if np.ndim(self.airmass) == 2:
                res = least_squares(lc2min_nneighbor, x0=[self.prior[k] for k in freekeys],
                                    bounds=[boundarray[:, 0], boundarray[:, 1]], jac='3-point', loss='linear')
            else:
                res = least_squares(lc2min_airmass, x0=[self.prior[k] for k in freekeys],
                                    bounds=[boundarray[:, 0], boundarray[:, 1]], jac='3-point', loss='linear')
        except Exception as e:
            print(f"{e} \nbounded light curve fitting failed...check priors "
                  "(e.g. estimated mid-transit time + orbital period)")

            for i, k in enumerate(freekeys):
                if not boundarray[i, 0] < self.prior[k] < boundarray[i, 1]:
                    print(f"bound: [{boundarray[i, 0]}, {boundarray[i, 1]}] prior: {self.prior[k]}")

            print("removing bounds and trying again...")

            if np.ndim(self.airmass) == 2:
                res = least_squares(lc2min_nneighbor, x0=[self.prior[k] for k in freekeys],
                                    method='lm', jac='3-point', loss='linear')
            else:
                res = least_squares(lc2min_airmass, x0=[self.prior[k] for k in freekeys],
                                    method='lm', jac='3-point', loss='linear')

        self.parameters = copy.deepcopy(self.prior)
        self.errors = {}

        for i, k in enumerate(freekeys):
            self.parameters[k] = res.x[i]
            self.errors[k] = 0

        self.create_fit_variables()

    def create_fit_variables(self):
        self.phase = get_phase(self.time, self.parameters['per'], self.parameters['tmid'])
        self.transit = transit(self.time, self.parameters)
        self.time_upsample = np.linspace(min(self.time), max(self.time), 1000)
        try:
            self.transit_upsample = np.asnumpy(transit(np.asnumpy(self.time_upsample), self.parameters))
            self.phase_upsample = get_phase(np.asnumpy(self.time_upsample), self.parameters['per'], self.parameters['tmid'])
        except AttributeError:
            self.transit_upsample = transit(self.time_upsample, self.parameters)
            self.phase_upsample = get_phase(self.time_upsample, self.parameters['per'], self.parameters['tmid'])
        if self.mode == "ns":
            try:
                self.parameters['a1'], self.errors['a1'] = mc_a1(np.array(self.parameters.get('a2', 0)), np.array(self.errors.get('a2', 1e-6)),
                                                             np.array(self.transit), np.array(self.airmass), np.array(self.data))
            except AttributeError:
                self.parameters['a1'], self.errors['a1'] = mc_a1(self.parameters.get('a2', 0), self.errors.get('a2', 1e-6),
                                                             self.transit, self.airmass, self.data)
        if np.ndim(self.airmass) == 2:
            try:
                detrended = np.asnumpy(np.array(self.data) / np.array(self.transit))
                self.wf = np.asnumpy(weightedflux(np.array(detrended), np.array(self.gw), np.array(self.nearest)))
                self.model = np.asnumpy(np.array(self.transit) * np.array(self.wf))
                self.detrended = np.asnumpy(np.array(self.data) / np.array(self.wf))
                self.detrendederr = np.asnumpy(np.array(self.dataerr) / np.array(self.wf))
            except AttributeError:
                detrended = self.data / self.transit
                self.wf = weightedflux(detrended, self.gw, self.nearest)
                self.model = self.transit * self.wf
                self.detrended = self.data / self.wf
                self.detrendederr = self.dataerr / self.wf
        else:

            try:
                self.airmass_model = np.asnumpy(self.parameters['a1'] * np.exp(self.parameters.get('a2', 0) * np.array(self.airmass)))
                self.model = np.asnumpy(np.array(self.transit) * np.array(self.airmass_model))
                self.detrended = np.asnumpy(np.array(self.data) / np.array(self.airmass_model))
                self.detrendederr = np.asnumpy(np.array(self.dataerr) / np.array(self.airmass_model))
            except AttributeError:
                self.airmass_model = self.parameters['a1'] * np.exp(self.parameters.get('a2', 0) * self.airmass)
                self.model = self.transit * self.airmass_model
                self.detrended = self.data / self.airmass_model
                self.detrendederr = self.dataerr / self.airmass_model

        try:
            self.residuals = np.asnumpy(np.array(self.data) - np.array(self.model))
            self.res_stdev = np.asnumpy(np.std(np.array(self.residuals))/np.median(np.array(self.data)))
            self.chi2 = np.asnumpy(np.sum(np.array(self.residuals) ** 2 / np.array(self.dataerr) ** 2))
            self.bic = np.asnumpy(len(self.bounds) * np.log(len(self.time)) - 2 * np.log(np.array(self.chi2)))
        except AttributeError:
            self.residuals = self.data - self.model
            self.res_stdev = np.std(self.residuals)/np.median(self.data)
            self.chi2 = np.sum(self.residuals ** 2 / self.dataerr ** 2)
            self.bic = len(self.bounds) * np.log(len(self.time)) - 2 * np.log(self.chi2)

        # compare fit chi2 to smoothed data chi2
        try:
            dt = np.diff(np.sort(np.array(self.time))).mean()
            si = np.argsort(np.array(self.time))
        except AttributeError:
            dt = np.diff(np.sort(self.time)).mean()
            si = np.argsort(self.time)
        try:
            self.sdata = savgol_filter(self.data[si], 1 + 2 * int(0.5 / 24 / dt), 2)
        except:
            self.sdata = np.ones(len(self.time))

        try:
            schi2 = np.sum((np.array(self.data)[si] - np.array(self.sdata)) ** 2 / np.array(self.dataerr)[si] ** 2)
            self.quality = np.asnumpy(schi2 / np.array(self.chi2))
        except AttributeError:
            schi2 = np.sum((self.data[si] - self.sdata) ** 2 / self.dataerr[si] ** 2)
            self.quality = schi2 / self.chi2

        # measured duration
        try:
            tdur = (np.array(self.transit) < 1).sum() * np.median(np.diff(np.sort(np.array(self.time))))
        except AttributeError:
            tdur = (self.transit < 1).sum() * np.median(np.diff(np.sort(self.time)))

        # test for partial transit
        try:
            newtime = np.asnumpy(np.linspace(np.array(self.parameters['tmid']) - 0.2, np.array(self.parameters['tmid']) + 0.2, 10000))
            newtran = transit(newtime, self.parameters)
            masktran = np.array(newtran) < 1
            newdur = np.asnumpy(np.diff(newtime).mean() * masktran.sum())
        except AttributeError:
            newtime = np.linspace(self.parameters['tmid'] - 0.2, self.parameters['tmid'] + 0.2, 10000)
            newtran = transit(newtime, self.parameters)
            masktran = newtran < 1
            newdur = np.diff(newtime).mean() * masktran.sum()

        self.duration_measured = tdur
        self.duration_expected = newdur

    def fit_nested(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.asnumpy(np.array([self.bounds[k] for k in freekeys]))
        bounddiff = np.asnumpy(np.diff(np.array(boundarray), 1).reshape(-1))

        # alloc data for best fit + error
        self.errors = {}
        self.quantiles = {}
        self.parameters = copy.deepcopy(self.prior)

        def loglike(pars):
            # chi-squared
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]
            model = transit(self.time, self.prior)
            try:
                model = np.asnumpy(np.array(model) * np.exp(self.prior['a2'].item() * np.array(self.airmass)))
                detrend = self.data / model  # used to estimate a1
                model = np.asnumpy(np.array(model) * np.median(np.array(detrend)))
                return -0.5 * np.sum(((np.array(self.data) - np.array(model)) / np.array(self.dataerr)) ** 2).item()
            except AttributeError:
                model *= np.exp(self.prior['a2'] * self.airmass)
                detrend = self.data / model  # used to estimate a1
                model *= np.median(detrend)
                return -0.5 * np.sum(((self.data - model) / self.dataerr) ** 2)

        def prior_transform(upars):
            # transform unit cube to prior volume
            return boundarray[:, 0] + bounddiff * upars

        try:
            self.ns_type = 'ultranest'
            test = ReactiveNestedSampler(freekeys, loglike, prior_transform)

            noop = lambda *args, **kwargs: None
            if self.verbose is True:
                self.results = test.run(max_ncalls=int(self.max_ncalls))
            else:
                self.results = test.run(max_ncalls=int(self.max_ncalls), show_status=False, viz_callback=noop)

            for i, key in enumerate(freekeys):
                self.parameters[key] = self.results['maximum_likelihood']['point'][i]
                self.errors[key] = self.results['posterior']['stdev'][i]
                self.quantiles[key] = [
                    self.results['posterior']['errlo'][i],
                    self.results['posterior']['errup'][i]]
        except NameError:
            self.ns_type = 'dynesty'
            dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=len(freekeys),
                                                    bound='multi', sample='unif')
            dsampler.run_nested(maxcall=int(1e5), dlogz_init=0.05,
                                maxbatch=10, nlive_batch=100, print_progressbool=self.verbose)
            self.results = dsampler.results

            tests = [copy.deepcopy(self.prior) for i in range(5)]

            # Derive kernel density estimate for best fit
            try:
                weights = np.exp(self.results.logwt - self.results.logz[-1])
            except AttributeError:
                weights = np.asnumpy(np.exp(np.array(self.results.logwt) - np.array(self.results.logz[-1])))
            samples = self.results['samples']
            logvol = self.results['logvol']
            wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
            try:
                logvol_grid = np.asnumpy(np.linspace(np.array(logvol[0]), np.array(logvol[-1]), 1000))  # resample
            except AttributeError:
                logvol_grid = np.linspace(logvol[0], logvol[-1], 1000)  # resample
            wt_grid = wt_kde.pdf(-logvol_grid)  # evaluate KDE PDF
            try:
                self.weights = np.asnumpy(np.interp(np.array(-logvol), np.array(-logvol_grid), np.array(wt_grid)))  # interpolate
            except AttributeError:
                self.weights = np.interp(-logvol, -logvol_grid, wt_grid)  # interpolate

            # errors + final values
            mean, cov = dynesty.utils.mean_and_cov(self.results.samples, weights)
            mean2, cov2 = dynesty.utils.mean_and_cov(self.results.samples, self.weights)
            for i in range(len(freekeys)):
                self.errors[freekeys[i]] = cov[i, i] ** 0.5
                tests[0][freekeys[i]] = mean[i]
                tests[1][freekeys[i]] = mean2[i]

                try:
                    counts, bins = np.histogram(np.array(samples[:, i]), bins=100, weights=weights)
                    mi = np.argmax(counts)
                    tests[4][freekeys[i]] = np.asnumpy(bins[mi] + 0.5 * np.mean(np.diff(bins)))
                except AttributeError:
                    counts, bins = np.histogram(samples[:, i], bins=100, weights=weights)
                    mi = np.argmax(counts)
                    tests[4][freekeys[i]] = bins[mi] + 0.5 * np.mean(np.diff(bins))

                # finds median and +- 2sigma, will vary from mode if non-gaussian
                self.quantiles[freekeys[i]] = dynesty.utils.quantile(self.results.samples[:, i], [0.025, 0.5, 0.975],
                                                                     weights=weights)
                tests[2][freekeys[i]] = self.quantiles[freekeys[i]][1]

            # find minimum near weighted mean
            mask = (samples[:, 0] < self.parameters[freekeys[0]] + 2 * self.errors[freekeys[0]]) & (
                    samples[:, 0] > self.parameters[freekeys[0]] - 2 * self.errors[freekeys[0]])
            bi = np.argmin(self.weights[mask])

            for i in range(len(freekeys)):
                tests[3][freekeys[i]] = samples[mask][bi, i]
                # tests[4][freekeys[i]] = np.average(samples[mask][:, i], weights=self.weights[mask], axis=0)

            # find best fit from chi2 minimization
            chis = []
            for i in range(len(tests)):
                lightcurve = transit(self.time, tests[i])
                try:
                    tests[i]['a1'] = np.asnumpy(mc_a1(np.array(tests[i]).get('a2', 0), np.array(self.errors).get('a2', 1e-6),
                                       np.array(lightcurve), np.array(self.airmass), np.array(self.data))[0])
                    airmass = np.asnumpy(np.array(tests[i]['a1']) * np.exp(np.array(tests[i]).get('a2', 0) * np.array(self.airmass)))
                    residuals = self.data - (lightcurve * airmass)
                    chis.append(np.sum(np.array(residuals) ** 2))
                except AttributeError:
                    tests[i]['a1'] = mc_a1(tests[i].get('a2', 0), self.errors.get('a2', 1e-6),
                                       lightcurve, self.airmass, self.data)[0]
                    airmass = tests[i]['a1'] * np.exp(tests[i].get('a2', 0) * self.airmass)
                    residuals = self.data - (lightcurve * airmass)
                    chis.append(np.sum(residuals ** 2))

            mi = np.argmin(chis)
            self.parameters = copy.deepcopy(tests[mi])

        # final model
        self.create_fit_variables()

    def plot_bestfit(self, title="", bin_dt=30. / (60 * 24), zoom=False, phase=True):
        f = plt.figure(figsize=(9, 6))
        f.subplots_adjust(top=0.92, bottom=0.09, left=0.14, right=0.98, hspace=0)
        ax_lc = plt.subplot2grid((4, 5), (0, 0), colspan=5, rowspan=3)
        ax_res = plt.subplot2grid((4, 5), (3, 0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        axs[0].set_title(title)
        axs[0].set_ylabel("Relative Flux", fontsize=14)
        axs[0].grid(True, ls='--')

        rprs2 = self.parameters['rprs'] ** 2
        rprs2err = 2 * self.parameters['rprs'] * self.errors['rprs']
        lclabel1 = r"$R^{2}_{p}/R^{2}_{s}$ = %s $\pm$ %s" % (
            str(round_to_2(rprs2, rprs2err)),
            str(round_to_2(rprs2err))
        )

        lclabel2 = r"$T_{mid}$ = %s $\pm$ %s BJD$_{TDB}$" % (
            str(round_to_2(self.parameters['tmid'], self.errors.get('tmid', 0))),
            str(round_to_2(self.errors.get('tmid', 0)))
        )

        lclabel = lclabel1 + "\n" + lclabel2

        if zoom:
            axs[0].set_ylim([1 - 1.25 * self.parameters['rprs'] ** 2, 1 + 0.5 * self.parameters['rprs'] ** 2])
        else:
            if phase:
                try:
                    axs[0].errorbar(self.phase, self.detrended, yerr=np.asnumpy(np.std(np.array(self.residuals)) / np.median(np.array(self.data))),
                                ls='none', marker='.', color='black', zorder=1, alpha=0.2)
                except AttributeError:
                    axs[0].errorbar(self.phase, self.detrended, yerr=np.std(self.residuals) / np.median(self.data),
                                ls='none', marker='.', color='black', zorder=1, alpha=0.2)
            else:
                try:
                    axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals) / np.median(self.data),
                                ls='none', marker='.', color='black', zorder=1, alpha=0.2)
                except AttributeError:
                    axs[0].errorbar(self.time, self.detrended, yerr=np.asnumpy(np.std(np.array(self.residuals)) / np.median(np.array(self.data))),
                                ls='none', marker='.', color='black', zorder=1, alpha=0.2)

        if phase:
            si = np.argsort(self.phase)
            try:
                bt2, br2, _ = time_bin(np.array(self.phase[si]) * np.array(self.parameters['per']),
                                   np.array(self.residuals[si]) / np.median(np.array(self.data)) * 1e2, bin_dt)
                axs[1].plot(self.phase, self.residuals / np.asnumpy(np.median(np.array(self.data))) * 1e2, 'k.', alpha=0.2,
                        label=r'$\sigma$ = {:.2f} %'.format(np.asnumpy(np.std(np.array(self.residuals) / np.median(np.array(self.data)) * 1e2))))
                axs[1].plot(np.asnumpy(bt2 / np.array(self.parameters['per'])), np.asnumpy(br2), 'bs', alpha=1, zorder=2)
            except AttributeError:
                bt2, br2, _ = time_bin(self.phase[si] * self.parameters['per'],
                                   self.residuals[si] / np.median(self.data) * 1e2, bin_dt)
                axs[1].plot(self.phase, self.residuals / np.median(self.data) * 1e2, 'k.', alpha=0.2,
                        label=r'$\sigma$ = {:.2f} %'.format(np.std(self.residuals / np.median(self.data) * 1e2)))
                axs[1].plot(bt2 / self.parameters['per'], br2, 'bs', alpha=1, zorder=2)
            axs[1].set_xlim([min(self.phase), max(self.phase)])
            axs[1].set_xlabel("Phase", fontsize=14)

            si = np.argsort(self.phase)
            try:
                bt2, bf2, bs = time_bin(np.array(self.phase[si]) * np.array(self.parameters['per']), np.array(self.detrended[si]), bin_dt)
                axs[0].errorbar(np.asnumpy(bt2 / np.array(self.parameters['per'])), np.asnumpy(bf2), yerr=np.asnumpy(bs), alpha=1, zorder=2, color='blue', ls='none',
                            marker='s')
            except AttributeError:
                bt2, bf2, bs = time_bin(self.phase[si] * self.parameters['per'], self.detrended[si], bin_dt)
                axs[0].errorbar(bt2 / self.parameters['per'], bf2, yerr=bs, alpha=1, zorder=2, color='blue', ls='none',
                            marker='s')
            # axs[0].plot(self.phase[si], self.transit[si], 'r-', zorder=3, label=lclabel)
            sii = np.argsort(self.phase_upsample)
            axs[0].plot(self.phase_upsample[sii], self.transit_upsample[sii], 'r-', zorder=3, label=lclabel)
            axs[0].set_xlim([min(self.phase), max(self.phase)])
            axs[0].set_xlabel("Phase ", fontsize=14)
        else:
            try:
                bt, br, _ = time_bin(np.array(self.time), np.array(self.residuals) / np.median(np.array(self.data)) * 1e2, bin_dt)
                axs[1].plot(np.asnumpy(self.time), np.asnumpy(self.residuals / np.median(np.array(self.data))) * 1e2, 'k.', alpha=0.2,
                        label=r'$\sigma$ = {:.2f} %'.format(np.asnumpy(np.std(np.array(self.residuals) / np.median(np.array(self.data)) * 1e2))))
                axs[1].plot(np.asnumpy(bt), np.asnumpy(br), 'bs', alpha=1, zorder=2, label=r'$\sigma$ = {:.2f} %'.format(np.asnumpy(np.std(br))))
            except AttributeError:
                bt, br, _ = time_bin(self.time, self.residuals / np.median(self.data) * 1e2, bin_dt)
                axs[1].plot(self.time, self.residuals / np.median(self.data) * 1e2, 'k.', alpha=0.2,
                        label=r'$\sigma$ = {:.2f} %'.format(np.std(self.residuals / np.median(self.data) * 1e2)))
                axs[1].plot(bt, br, 'bs', alpha=1, zorder=2, label=r'$\sigma$ = {:.2f} %'.format(np.std(br)))
            axs[1].set_xlim([min(self.time), max(self.time)])
            axs[1].set_xlabel("Time [day]", fontsize=14)

            try:
                bt, bf, bs = time_bin(np.array(self.time), np.array(self.detrended), bin_dt)
                si = np.argsort(np.array(self.time))
                sii = np.argsort(np.array(self.time_upsample))
                axs[0].errorbar(np.asnumpy(bt), np.asnumpy(bf), yerr=np.asnumpy(bs), alpha=1, zorder=2, color='blue', ls='none', marker='s')
                axs[0].plot(np.asnumpy(self.time_upsample[sii]), np.asnumpy(self.transit_upsample[sii]), 'r-', zorder=3, label=lclabel)
            except AttributeError:
                bt, bf, bs = time_bin(self.time, self.detrended, bin_dt)
                si = np.argsort(self.time)
                sii = np.argsort(self.time_upsample)
                axs[0].errorbar(bt, bf, yerr=bs, alpha=1, zorder=2, color='blue', ls='none', marker='s')
                axs[0].plot(self.time_upsample[sii], self.transit_upsample[sii], 'r-', zorder=3, label=lclabel)
            axs[0].set_xlim([min(self.time), max(self.time)])
            axs[0].set_xlabel("Time [day]", fontsize=14)

        axs[0].get_xaxis().set_visible(False)
        axs[1].legend(loc='best')
        axs[0].legend(loc='best')
        axs[1].set_ylabel("Residuals [%]", fontsize=14)
        axs[1].grid(True, ls='--', axis='y')
        return f, axs

    def plot_triangle(self):
        if self.ns_type == 'ultranest':
            ranges = []
            try:
                mask1 = np.asnumpy(np.ones(len(self.results['weighted_samples']['logl']), dtype=bool))
                mask2 = np.asnumpy(np.ones(len(self.results['weighted_samples']['logl']), dtype=bool))
                mask3 = np.asnumpy(np.ones(len(self.results['weighted_samples']['logl']), dtype=bool))
            except AttributeError:
                mask1 = np.ones(len(self.results['weighted_samples']['logl']), dtype=bool)
                mask2 = np.ones(len(self.results['weighted_samples']['logl']), dtype=bool)
                mask3 = np.ones(len(self.results['weighted_samples']['logl']), dtype=bool)
            titles = []
            labels = []
            flabels = {
                'rprs': r'R$_{p}$/R$_{s}$',
                'per': r'Period [day]',
                'tmid': r'T$_{mid}$',
                'ars': r'a/R$_{s}$',
                'inc': r'Inc. [deg]',
                'u1': r'u$_1$',
                'fpfs': r'F$_{p}$/F$_{s}$',
                'omega': r'$\omega$ [deg]',
                'mplanet': r'M$_{p}$ [M$_{\oplus}$]',
                'mstar': r'M$_{s}$ [M$_{\odot}$]',
                'ecc': r'$e$',
                'c0': r'$c_0$',
                'c1': r'$c_1$',
                'c2': r'$c_2$',
                'c3': r'$c_3$',
                'c4': r'$c_4$',
                'a0': r'$a_0$',
                'a1': r'$a_1$',
                'a2': r'$a_2$'
            }
            for i, key in enumerate(self.quantiles):
                labels.append(flabels.get(key, key))
                titles.append(f"{self.parameters[key]:.5f} +- {self.errors[key]:.5f}")
                ranges.append([
                    self.parameters[key] - 5 * self.errors[key],
                    self.parameters[key] + 5 * self.errors[key]
                ])

                if key == 'a2' or key == 'a1':
                    continue

                mask3 = mask3 & \
                    (self.results['weighted_samples']['points'][:, i] > (self.parameters[key] - 3 * self.errors[key])) & \
                    (self.results['weighted_samples']['points'][:, i] < (self.parameters[key] + 3 * self.errors[key]))

                mask1 = mask1 & \
                    (self.results['weighted_samples']['points'][:, i] > (self.parameters[key] - self.errors[key])) & \
                    (self.results['weighted_samples']['points'][:, i] < (self.parameters[key] + self.errors[key]))

                mask2 = mask2 & \
                    (self.results['weighted_samples']['points'][:, i] > (self.parameters[key] - 2 * self.errors[key])) & \
                    (self.results['weighted_samples']['points'][:, i] < (self.parameters[key] + 2 * self.errors[key]))

            chi2 = self.results['weighted_samples']['logl'] * -2
            try:
                fig = corner(self.results['weighted_samples']['points'],
                         labels=labels,
                         bins=int(np.asnumpy(np.sqrt(np.array(self.results['samples']).shape[0]))),
                         range=ranges,
                         # quantiles=(0.1, 0.84),
                         plot_contours=True,
                         levels=[np.asnumpy(np.percentile(np.array(chi2[mask1]), 95)), np.asnumpy(np.percentile(np.array(chi2[mask2]), 95)),
                                 np.asnumpy(np.percentile(np.array(chi2[mask3]), 95))],
                         plot_density=False,
                         titles=titles,
                         data_kwargs={
                             'c': chi2,
                             'vmin': np.asnumpy(np.percentile(np.array(chi2[mask3]), 1)),
                             'vmax': np.asnumpy(np.percentile(np.array(chi2[mask3]), 95)),
                             'cmap': 'viridis'
                         },
                         label_kwargs={
                             'labelpad': 15,
                         },
                         hist_kwargs={
                             'color': 'black',
                         }
                         )
            except AttributeError:
                fig = corner(self.results['weighted_samples']['points'],
                         labels=labels,
                         bins=int(np.sqrt(self.results['samples'].shape[0])),
                         range=ranges,
                         # quantiles=(0.1, 0.84),
                         plot_contours=True,
                         levels=[np.percentile(chi2[mask1], 95), np.percentile(chi2[mask2], 95),
                                 np.percentile(chi2[mask3], 95)],
                         plot_density=False,
                         titles=titles,
                         data_kwargs={
                             'c': chi2,
                             'vmin': np.percentile(chi2[mask3], 1),
                             'vmax': np.percentile(chi2[mask3], 95),
                             'cmap': 'viridis'
                         },
                         label_kwargs={
                             'labelpad': 15,
                         },
                         hist_kwargs={
                             'color': 'black',
                         }
                         )
        else:
            fig, axs = dynesty.plotting.cornerplot(self.results, labels=list(self.bounds.keys()),
                                                   quantiles_2d=[0.4, 0.85],
                                                   smooth=0.015, show_titles=True, use_math_text=True, title_fmt='.2e',
                                                   hist2d_kwargs={ 'fill_contours': False})
            dynesty.plotting.cornerpoints(self.results, labels=list(self.bounds.keys()),
                                          fig=[fig, axs[1:, :-1]], plot_kwargs={'alpha': 0.1, 'zorder': 1, })
        return fig

# simultaneously fit multiple data sets with global and local parameters
class glc_fitter(lc_fitter):
    # needed for lc_fitter
    ns_type = 'ultranest'

    def __init__(self, input_data, global_bounds, local_bounds, individual_fit=False, stdev_cutoff=0.03, verbose=False):
        # keys for input_data: time, flux, ferr, airmass, priors all numpy arrays
        self.lc_data = copy.deepcopy(input_data)
        self.global_bounds = global_bounds
        self.local_bounds = local_bounds
        self.individual_fit = individual_fit
        self.stdev_cutoff = stdev_cutoff
        self.verbose = verbose
        self.results = None
        self.fit_nested()

    def fit_nested(self):

        # create bound arrays for generating samples
        nobs = len(self.lc_data)
        gfreekeys = list(self.global_bounds.keys())

        # if isinstance(self.local_bounds, dict):
        #     lfreekeys = list(self.local_bounds.keys())
        #     boundarray = np.vstack([ [self.global_bounds[k] for k in gfreekeys], [self.local_bounds[k] for k in lfreekeys]*nobs ])
        # else:
        #     # if list type
        lfreekeys = []
        boundarray = [self.global_bounds[k] for k in gfreekeys]
        for i in range(nobs):
            lfreekeys.append(list(self.local_bounds[i].keys()))
            boundarray.extend([self.local_bounds[i][k] for k in lfreekeys[-1]])
            #print((list(self.local_bounds[i].keys()), [self.local_bounds[i][k] for k in lfreekeys[-1]]))
        try:
            boundarray = np.asnumpy(np.array(boundarray, dtype=np.float64))
        except AttributeError:
            boundarray = np.array(boundarray)
        print(boundarray)

        # fit individual light curves to constrain priors
        if self.individual_fit:
            for i in range(nobs):

                print(f"Fitting individual light curve {i+1}/{nobs}")
                try:
                    mybounds = dict(**self.local_bounds[i], **self.global_bounds)
                except:
                    mybounds = {}
                    for k in self.local_bounds[i]:
                        mybounds[k] = self.local_bounds[i][k]
                    for k in self.global_bounds:
                        mybounds[k] = self.global_bounds[k]
                if 'per' in mybounds: del(mybounds['per'])
                if 'inc' in mybounds and 'rprs' in mybounds: del(mybounds['inc'])
                if 'tmid' in mybounds:
                    # find the closet mid transit time to the last observation
                    phase = (self.lc_data[i]['time'][-1] - self.lc_data[i]['priors']['tmid']) / self.lc_data[i]['priors']['per']
                    try:
                        nepochs = np.asnumpy(np.round(np.array(phase, dtype=np.float64)))
                    except AttributeError:
                        nepochs = np.round(phase)
                    newtmid = self.lc_data[i]['priors']['tmid'] + nepochs * self.lc_data[i]['priors']['per']
                    try:
                        err = np.asnumpy(np.diff(np.array(mybounds['tmid'], dtype=np.float64))[0]/2.)
                    except AttributeError:
                        err = np.diff(mybounds['tmid'])[0]/2.
                    mybounds['tmid'] = [newtmid - err, newtmid + err ]

                # fit individual light curve
                myfit = lc_fitter(
                    self.lc_data[i]['time'],
                    self.lc_data[i]['flux'],
                    self.lc_data[i]['ferr'],
                    self.lc_data[i]['airmass'],
                    self.lc_data[i]['priors'],
                    mybounds
                )

                # check stdev_cutoff and residuals
                if myfit.res_stdev > self.stdev_cutoff:
                    print(f"WARNING: Stdev of residuals is large! {myfit.res_stdev:.3f} > {self.stdev_cutoff:.3f}")
                    #raise ValueError(f"Stdev of residuals is too large, please remove id: {self.lc_data[i]['name']}

                # copy data over for individual fits
                self.lc_data[i]['individual'] = myfit.parameters.copy()
                self.lc_data[i]['individual_err'] = myfit.errors.copy()
                self.lc_data[i]['res_stdev'] = myfit.res_stdev
                self.lc_data[i]['quality'] = myfit.quality

                ti = sum([len(self.local_bounds[k]) for k in range(i)])
                # update local priors
                for j, key in enumerate(self.local_bounds[i].keys()):

                    boundarray[j+ti+len(gfreekeys),0] = myfit.parameters[key] - 5*myfit.errors[key]
                    boundarray[j+ti+len(gfreekeys),1] = myfit.parameters[key] + 5*myfit.errors[key]

                    if key == 'rprs':
                        boundarray[j+ti+len(gfreekeys),0] = max(0,myfit.parameters[key] - 5*myfit.errors[key])

                # print name and stdev of residuals
                try:
                    mint = np.min(np.array(self.lc_data[i]['time'], dtype=np.float64))
                    maxt = np.max(np.array(self.lc_data[i]['time'], dtype=np.float64))
                except AttributeError:
                    mint = np.min(self.lc_data[i]['time'])
                    maxt = np.max(self.lc_data[i]['time'])
                try:
                    try:
                        print(f"{self.lc_data[i]['name']} & {Time(mint,format='jd').isot} & {Time(maxt,format='jd').isot} & {np.asnumpy(np.std(np.array(myfit.residuals, dtype=np.float64)))} & {len(self.lc_data[i]['time'])}")
                    except AttributeError:
                        print(f"{self.lc_data[i]['name']} & {Time(mint,format='jd').isot} & {Time(maxt,format='jd').isot} & {np.std(myfit.residuals)} & {len(self.lc_data[i]['time'])}")
                except:
                    try:
                        print(f"{self.lc_data[i]['name']} & {mint} & {maxt} & {np.asnumpy(np.std(np.array(myfit.residuals, dtype=np.float64)))} & {len(self.lc_data[i]['time'])}")
                    except AttributeError:
                        print(f"{self.lc_data[i]['name']} & {mint} & {maxt} & {np.std(myfit.residuals)} & {len(self.lc_data[i]['time'])}")

                del(myfit)

        # transform unit cube to prior volume
        try:
            bounddiff = np.asnumpy(np.diff(np.array(boundarray, dtype=np.float64),1).reshape(-1))
        except AttributeError:
            bounddiff = np.diff(boundarray,1).reshape(-1)
        print(bounddiff)
        try:
            #jax.device_put(boundarray)
            #jax.device_put(bounddiff)
            boundarray = jnp.array(boundarray, dtype=jnp.float64)
            bounddiff = jnp.array(bounddiff, dtype=jnp.float64)
        except NameError:
            pass
        def prior_transform(upars): # this runs on GPU via JAX arrays
            #try:
            #    #print(jax.dlpack.from_dlpack(np.array(boundarray[:,0], dtype=np.float64)))
            #    #print(jax.dlpack.from_dlpack(np.array(bounddiff, dtype=np.float64)))
            #    return (jax.dlpack.from_dlpack(np.array(boundarray[:,0], dtype=np.float64)) + jax.dlpack.from_dlpack(np.array(bounddiff, dtype=np.float64))*upars)
            #except NameError:
            return (boundarray[:,0] + bounddiff*upars)

        def compute_chi2(limb_darkening_coefficients, rprs, per, ars, ecc, inc, omega, tmid, times, a2, airmass, flux, ferr):
            chi2 = 0

            # compute model
            #print(self.lc_data[i]['time'])
            #print(self.lc_data[i]['priors'])
            try:
                #dlpack = np.array(self.lc_data[i]['time'], dtype=np.float64).toDlpack()
                #model = transit(jax.dlpack.from_dlpack(dlpack), self.lc_data[i]['priors'])
                #del dlpack
                #model = transit(jnp.array(self.lc_data[i]['time'], dtype=jnp.float64), self.lc_data[i]['priors'])
                mmodel = pytransit(limb_darkening_coefficients, rprs, per, ars, ecc, inc, omega, tmid, times, 
                      #method='claret', 
                      precision=3)
            except NameError:
                model = transit(self.lc_data[i]['time'], self.lc_data[i]['priors'])
            #print(model)
            #try:
            #    model = np.asnumpy(np.array(model, dtype=np.float64) * np.exp(np.array(self.lc_data[i]['priors']['a2'], dtype=np.float64)*np.array(self.lc_data[i]['airmass'], dtype=np.float64)))
            #except AttributeError:
            #    model *= np.exp(self.lc_data[i]['priors']['a2']*self.lc_data[i]['airmass'])
            #model *= np.exp(np.array(self.lc_data[i]['priors']['a2'], dtype=np.float64)*np.array(self.lc_data[i]['airmass'], dtype=np.float64))
            try:
                #model *= jnp.exp(self.lc_data[i]['priors']['a2']*np.array(self.lc_data[i]['airmass'], dtype=np.float64))
                model *= a2*airmass
            except NameError:
                model *= np.exp(self.lc_data[i]['priors']['a2']*self.lc_data[i]['airmass'])
            #print(model)
            #detrend = self.lc_data[i]['flux']/model
            #detrend = np.array(self.lc_data[i]['flux'], dtype=np.float64)/model
            try:
                #detrend = jnp.array(self.lc_data[i]['flux'], dtype=jnp.float64)/model
                detrend = flux/model
            except NameError:
                detrend = self.lc_data[i]['flux']/model
            #print(detrend)
            #try:
            #    model = np.asnumpy(np.array(model, dtype=np.float64) * np.mean(np.array(detrend, dtype=np.float64)))
            #except AttributeError:
            #    model *= np.mean(detrend)
            #model = np.array(model, dtype=np.float64) * np.mean(np.array(detrend, dtype=np.float64))
            try:
                model *= jnp.mean(detrend)
            except NameError:
                model *= np.mean(detrend)
            #print(model)

            # add to chi2
            #try:
            #    #chi2 += np.sum( ((np.array(self.lc_data[i]['flux'], dtype=np.float64)-np.array(model, dtype=np.float64))/np.array(self.lc_data[i]['ferr'], dtype=np.float64))**2 ).item()
            #    chi2 += np.sum( ((np.array(self.lc_data[i]['flux'], dtype=np.float64)-model)/np.array(self.lc_data[i]['ferr'], dtype=np.float64))**2 ).item()
            #except AttributeError:
            #    chi2 += np.sum( ((self.lc_data[i]['flux']-model)/self.lc_data[i]['ferr'])**2 )
            try:
                #chi2 += jnp.sum( ((jnp.array(self.lc_data[i]['flux'], dtype=jnp.float64)-model)/jnp.array(self.lc_data[i]['ferr'], dtype=jnp.float64))**2 ).item()
                chi2 += jnp.sum( ((flux-model)/ferr)**2 )
            except NameError:
                chi2 += np.sum( ((self.lc_data[i]['flux']-model)/self.lc_data[i]['ferr'])**2 )
            print(chi2)
                    
            return chi2

        def loglike(pars): # this runs on GPU via JAX arrays, but manipulates only Cupy arrays internally
            chi2 = 0

            #print(nobs)
            #print(pars.shape)
            #print(type(pars))
            #print(jnp.tile(pars, nobs).shape)
            #print(type(jnp.tile(pars, nobs)))
            #print(jax.vmap(compute_chi2, axis_size=nobs, axis_name='i')(jnp.tile(pars, nobs)))
            #print(jnp.sum(jax.vmap(compute_chi2, axis_size=nobs, axis_name='i')(jnp.tile(pars, nobs))))
            #print(jnp.sum(jax.vmap(compute_chi2, axis_size=nobs, axis_name='i')(jnp.tile(pars, nobs))).item())
            try:
                # make global time array and masks for each data set
                alltime = []
                for i in range(nobs):
                    alltime.extend(self.lc_data[i]['time'])

                limb_darkening_coefficients = np.array([[]], dtype=np.float64).reshape(0,4)
                rprs = np.array([], dtype=np.float64)
                per = np.array([], dtype=np.float64)
                ars = np.array([], dtype=np.float64)
                ecc = np.array([], dtype=np.float64)
                inc = np.array([], dtype=np.float64)
                omega = np.array([], dtype=np.float64)
                tmid = np.array([], dtype=np.float64)
                a2 = np.array([], dtype=np.float64)

                times = np.array([[]], dtype=np.float64).reshape(0,len(alltime))
                flux = np.array([[]], dtype=np.float64).reshape(0,len(alltime))
                ferr = np.array([[]], dtype=np.float64).reshape(0,len(alltime))
                airmass = np.array([[]], dtype=np.float64).reshape(0,len(alltime))

                # for each light curve
                for i in range(nobs):
                    #print(i)
                    # global keys
                    for j, key in enumerate(gfreekeys):
                        #print((j,key))
                        try:
                            dlpack = pars[j].toDlpack()
                            #self.lc_data[i]['priors'][key] = np.asnumpy(np.from_dlpack(dlpack))
                            self.lc_data[i]['priors'][key] = np.from_dlpack(dlpack).item()
                            #self.lc_data[i]['priors'][key] = np.from_dlpack(dlpack)
                            del dlpack
                        except AttributeError:
                            #self.lc_data[i]['priors'][key] = np.array(pars[j], dtype=np.float64)
                            self.lc_data[i]['priors'][key] = pars[j]
                    

                    # local keys
                    ti = sum([len(self.local_bounds[k]) for k in range(i)])
                    for j, key in enumerate(lfreekeys[i]):
                        #print((j,key))
                        try:
                            #self.lc_data[i]['priors'][key] = np.asnumpy(np.from_dlpack(pars[j+ti+len(gfreekeys)]))
                            dlpack = pars[j+ti+len(gfreekeys)].toDlpack()
                            self.lc_data[i]['priors'][key] = np.from_dlpack(dlpack).item()
                            #self.lc_data[i]['priors'][key] = np.from_dlpack(dlpack)
                            del dlpack
                        except AttributeError:
                            #self.lc_data[i]['priors'][key] = np.array(pars[j+ti+len(gfreekeys)], dtype=np.float64)
                            self.lc_data[i]['priors'][key] = pars[j+ti+len(gfreekeys)]

                    limb_darkening_coefficients = np.append(limb_darkening_coefficients, 
                        np.array([[self.lc_data[i]['priors']['u0'], 
                                   self.lc_data[i]['priors']['u1'], 
                                   self.lc_data[i]['priors']['u2'], 
                                   self.lc_data[i]['priors']['u3']]], dtype=np.float64), axis=0)
                    rprs = np.append(rprs, self.lc_data[i]['priors']['rprs'])
                    per = np.append(per, self.lc_data[i]['priors']['per'])
                    ars = np.append(ars, self.lc_data[i]['priors']['ars'])
                    ecc = np.append(ecc, self.lc_data[i]['priors']['ecc'])
                    inc = np.append(inc, self.lc_data[i]['priors']['inc'])
                    omega = np.append(omega, self.lc_data[i]['priors']['omega'])
                    tmid = np.append(tmid, self.lc_data[i]['priors']['tmid'])
                    a2 = np.append(a2, self.lc_data[i]['priors']['a2'])
                    

                    empty_array = np.full_like(np.array(alltime), fill_value=np.NaN, dtype=np.float64)
                    _, ind, _ = np.intersect1d(np.array(alltime), self.lc_data[i]['time'], return_indices=True, assume_unique=True)
                    time_array = empty_array
                    np.put(time_array, ind, self.lc_data[i]['time'])
                    times = np.append(times, [time_array], axis=0)
                    #tmask = np.in1d(np.array(alltime), self.data[i]['time'])
                    #times = np.append(time, [np.ma.array(alltime, mask=tmask)], axis=0)
                    flux_array = empty_array
                    np.put(flux_array, ind, self.lc_data[i]['flux'])
                    flux = np.append(flux, [flux_array], axis=0)
                    ferr_array = empty_array
                    np.put(ferr_array, ind, self.lc_data[i]['ferr'])
                    ferr = np.append(ferr, [ferr_array], axis=0)
                    airmass_array = empty_array
                    np.put(airmass_array, ind, self.lc_data[i]['airmass'])
                    airmass = np.append(airmass, [airmass_array], axis=0)

                print(nobs)
                print(limb_darkening_coefficients.shape)
                print(rprs.shape)
                print(per.shape)
                print(ars.shape)
                print(ecc.shape)
                print(inc.shape)
                print(omega.shape)
                print(tmid.shape)
                print(times.shape)
                print(a2.shape)
                print(airmass.shape)
                print(flux.shape)
                print(ferr.shape)

                try:
                    #chi2 = jnp.sum(jax.pmap(compute_chi2, axis_size=nobs, axis_name='i')(jax.tile(pars, nobs))).item()
                    #chi2 = jnp.sum(jax.pmap(compute_chi2, axis_size=nobs, axis_name='i')(pars.tile(nobs))).item()
                    #chi2 = jnp.sum(jax.pmap(compute_chi2, axis_size=nobs, axis_name='i')(jnp.tile(pars, nobs),jnp.arange(0, nobs, 1, dtype=jnp.int))).item()
                    chi2 = jnp.sum(jax.pmap(compute_chi2, axis_size=nobs, axis_name='i')(
                        limb_darkening_coefficients, rprs, per, ars, ecc, inc, omega, tmid, 
                        times, a2, airmass, flux, ferr
                        )).item()
                except ValueError:
                    #chi2 = jnp.sum(jax.vmap(compute_chi2, axis_size=nobs, axis_name='i')(jax.tile(pars, nobs))).item()
                    #chi2 = jnp.sum(jax.vmap(compute_chi2, axis_size=nobs, axis_name='i')(pars.tile(nobs))).item()
                    #chi2 = jnp.sum(jax.vmap(compute_chi2, axis_size=nobs, axis_name='i')(jnp.tile(pars, nobs),jnp.arange(0, nobs, 1, dtype=jnp.int))).item()
                    chi2 = jnp.sum(jax.vmap(compute_chi2, axis_size=nobs, axis_name='i')(
                        limb_darkening_coefficients, rprs, per, ars, ecc, inc, omega, tmid, 
                        times, a2, airmass, flux, ferr
                        )).item()

                # maximization metric for nested sampling
                return -0.5*chi2
            except NameError:
                # for each light curve
                for i in prange(nobs): # Parallelization using Numba
                    #print(i)
                    # global keys
                    for j, key in enumerate(gfreekeys):
                        #print((j,key))
                        try:
                            dlpack = pars[j].toDlpack()
                            #self.lc_data[i]['priors'][key] = np.asnumpy(np.from_dlpack(dlpack))
                            self.lc_data[i]['priors'][key] = np.from_dlpack(dlpack).item()
                            #self.lc_data[i]['priors'][key] = np.from_dlpack(dlpack)
                            del dlpack
                        except AttributeError:
                            #self.lc_data[i]['priors'][key] = np.array(pars[j], dtype=np.float64)
                            self.lc_data[i]['priors'][key] = pars[j]
                    

                    # local keys
                    ti = sum([len(self.local_bounds[k]) for k in range(i)])
                    for j, key in enumerate(lfreekeys[i]):
                        #print((j,key))
                        try:
                            #self.lc_data[i]['priors'][key] = np.asnumpy(np.from_dlpack(pars[j+ti+len(gfreekeys)]))
                            dlpack = pars[j+ti+len(gfreekeys)].toDlpack()
                            self.lc_data[i]['priors'][key] = np.from_dlpack(dlpack).item()
                            #self.lc_data[i]['priors'][key] = np.from_dlpack(dlpack)
                            del dlpack
                        except AttributeError:
                            #self.lc_data[i]['priors'][key] = np.array(pars[j+ti+len(gfreekeys)], dtype=np.float64)
                            self.lc_data[i]['priors'][key] = pars[j+ti+len(gfreekeys)]

                    # compute model
                    #print(self.lc_data[i]['time'])
                    #print(self.lc_data[i]['priors'])
                    try:
                        #dlpack = np.array(self.lc_data[i]['time'], dtype=np.float64).toDlpack()
                        #model = transit(jax.dlpack.from_dlpack(dlpack), self.lc_data[i]['priors'])
                        #del dlpack
                        model = transit(jnp.array(self.lc_data[i]['time'], dtype=jnp.float64), self.lc_data[i]['priors'])
                    except NameError:
                        model = transit(self.lc_data[i]['time'], self.lc_data[i]['priors'])
                    #print(model)
                    #try:
                    #    model = np.asnumpy(np.array(model, dtype=np.float64) * np.exp(np.array(self.lc_data[i]['priors']['a2'], dtype=np.float64)*np.array(self.lc_data[i]['airmass'], dtype=np.float64)))
                    #except AttributeError:
                    #    model *= np.exp(self.lc_data[i]['priors']['a2']*self.lc_data[i]['airmass'])
                    #model *= np.exp(np.array(self.lc_data[i]['priors']['a2'], dtype=np.float64)*np.array(self.lc_data[i]['airmass'], dtype=np.float64))
                    try:
                        model *= jnp.exp(self.lc_data[i]['priors']['a2']*np.array(self.lc_data[i]['airmass'], dtype=np.float64))
                    except NameError:
                        model *= np.exp(self.lc_data[i]['priors']['a2']*self.lc_data[i]['airmass'])
                    #print(model)
                    #detrend = self.lc_data[i]['flux']/model
                    #detrend = np.array(self.lc_data[i]['flux'], dtype=np.float64)/model
                    try:
                        detrend = jnp.array(self.lc_data[i]['flux'], dtype=jnp.float64)/model
                    except NameError:
                        detrend = self.lc_data[i]['flux']/model
                    #print(detrend)
                    #try:
                    #    model = np.asnumpy(np.array(model, dtype=np.float64) * np.mean(np.array(detrend, dtype=np.float64)))
                    #except AttributeError:
                    #    model *= np.mean(detrend)
                    #model = np.array(model, dtype=np.float64) * np.mean(np.array(detrend, dtype=np.float64))
                    try:
                        model *= jnp.mean(detrend)
                    except NameError:
                        model *= np.mean(detrend)
                    #print(model)

                    # add to chi2
                    #try:
                    #    #chi2 += np.sum( ((np.array(self.lc_data[i]['flux'], dtype=np.float64)-np.array(model, dtype=np.float64))/np.array(self.lc_data[i]['ferr'], dtype=np.float64))**2 ).item()
                    #    chi2 += np.sum( ((np.array(self.lc_data[i]['flux'], dtype=np.float64)-model)/np.array(self.lc_data[i]['ferr'], dtype=np.float64))**2 ).item()
                    #except AttributeError:
                    #    chi2 += np.sum( ((self.lc_data[i]['flux']-model)/self.lc_data[i]['ferr'])**2 )
                    try:
                        chi2 += jnp.sum( ((jnp.array(self.lc_data[i]['flux'], dtype=jnp.float64)-model)/jnp.array(self.lc_data[i]['ferr'], dtype=jnp.float64))**2 ).item()
                    except NameError:
                        chi2 += np.sum( ((self.lc_data[i]['flux']-model)/self.lc_data[i]['ferr'])**2 )
                    #print(chi2)

                # maximization metric for nested sampling
                return -0.5*chi2

        freekeys = []+gfreekeys
        for n in range(nobs):
            for k in lfreekeys[n]:
                #clean_name = self.lc_data[n].get('name', n).replace(' ','_').replace('(','').replace(')','').replace('[','').replace(']','').replace('-','_').split('-')[0]
                freekeys.append(f"local_{k}_{n}")

        # for each light curve
        for i in range(nobs): 
            self.lc_data[i]['time'] = np.array(self.lc_data[i]['time'], dtype=np.float64)
            for k in self.lc_data[i]['priors'].keys():
            #    self.lc_data[i]['priors'][k] = np.array(self.lc_data[i]['priors'][k], dtype=np.float64)
                try:
                    self.lc_data[i]['priors'][k] = self.lc_data[i]['priors'][k].item()
                except AttributeError:
                    pass
        #try:
        #    jax.device_put(self.lc_data)
        #    jax.device_put(gauss_table)
        #except NameError:
        #    pass

        noop = lambda *args, **kwargs: None
        if self.verbose:
            sampler = ReactiveNestedSampler(freekeys, loglike, prior_transform)
            nsteps = 2 * len(freekeys)
            #sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps,generate_direction=ultranest.stepsampler.generate_mixture_random_direction)
            #sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps,generate_direction=ultranest.stepsampler.generate_cube_oriented_direction)
            try:
                #sampler.stepsampler = ultranest.popstepsampler.PopulationRandomWalkSampler(popsize=40,nsteps=nsteps,generate_direction=ultranest.popstepsampler.generate_region_random_direction)
                sampler.stepsampler = ultranest.popstepsampler.PopulationSliceSampler(popsize=40,nsteps=nsteps,generate_direction=ultranest.popstepsampler.generate_cube_oriented_direction)
                self.results = np.asnumpy(np.from_dlpack(sampler.run(max_ncalls=2e6, show_status=True))) # pached
            except AttributeError:
                #sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps,generate_direction=ultranest.stepsampler.generate_mixture_random_direction)
                sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps,generate_direction=ultranest.stepsampler.generate_cube_oriented_direction)
                self.results = sampler.run(max_ncalls=2e6, show_status=True) # pached
        else:
            self.results = ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=1e6, show_status=False, viz_callback=noop)

        #try:
        #    jax.device_get(self.lc_data)
        #    jax.device_get(self.results)
        #except NameError:
        #    pass
        # for each light curve
        for i in range(nobs): 
            try:
                self.lc_data[i]['time'] = np.asnumpy(self.lc_data[i]['time'])       
            except AttributeError:
                pass
        # for each light curve
        for i in range(nobs):
            for k in self.lc_data[i]['priors'].keys():
                try:
                    self.lc_data[i]['priors'][k] = np.asnumpy(self.lc_data[i]['priors']).item()
                except AttributeError:
                    pass

        self.quantiles = {}
        self.errors = {}
        try:
            self.parameters = np.asnumpy(self.lc_data[0]['priors']).copy()
        except AttributeError:
            self.parameters = self.lc_data[0]['priors'].copy()

        for i, key in enumerate(freekeys):
            self.parameters[key] = self.results['maximum_likelihood']['point'][i]
            #self.errors[key] = self.results['posterior']['median'][i]

            self.errors[key] = self.results['posterior']['stdev'][i]
            self.quantiles[key] = [
                self.results['posterior']['errlo'][i],
                self.results['posterior']['errup'][i]]

        # create an average Rp/Rs if it is not in global keys
        # check if 'rprs' is in lfreekeys
        rprs_in_local = False
        for i in range(nobs):
            if 'rprs' in lfreekeys[i]:
                rprs_in_local = True
                break

        if rprs_in_local:
            local_rprs = [] # used for creating an average value
            local_rprs_err = []

        # loop over observations
        for n in range(nobs):
            self.lc_data[n]['errors'] = {}
            
            # copy global parameters
            self.lc_data[n]['priors'] = copy.deepcopy(self.parameters)

            # loop over local keys and save best fit values
            for k in lfreekeys[n]:

                # create key to get results
                pkey = f"local_{k}_{n}"
                
                # overwrite priors with best fit value
                self.lc_data[n]['priors'][k] = self.parameters[pkey]
                self.lc_data[n]['errors'][k] = self.errors[pkey]

                # update key for final bestfit plot if needed
                if k == 'rprs':
                    local_rprs.append(self.lc_data[n]['priors'][k])
                    local_rprs_err.append(self.lc_data[n]['errors'][k])

            # solve for a1
            try:
                model = np.asnumpy(transit(self.lc_data[n]['time'], self.lc_data[n]['priors']))
            except AttributeError:
                model = transit(self.lc_data[n]['time'], self.lc_data[n]['priors'])
            try:
                airmass = np.asnumpy(np.exp(np.array(self.lc_data[n]['airmass'], dtype=np.float64)*np.array(self.lc_data[n]['priors']['a2'], dtype=np.float64)))
            except AttributeError:
                airmass = np.exp(self.lc_data[n]['airmass']*self.lc_data[n]['priors']['a2'])
            detrend = self.lc_data[n]['flux']/(model*airmass)
            try:
                self.lc_data[n]['priors']['a1'] = np.asnumpy(np.mean(np.array(detrend, dtype=np.float64)))
            except AttributeError:
                self.lc_data[n]['priors']['a1'] = np.mean(detrend)
            self.lc_data[n]['residuals'] = self.lc_data[n]['flux'] - model*airmass*self.lc_data[n]['priors']['a1']
            self.lc_data[n]['detrend'] = self.lc_data[n]['flux']/(airmass*self.lc_data[n]['priors']['a1'])

            # phase
            self.lc_data[n]['phase'] = get_phase(self.lc_data[n]['time'], self.lc_data[n]['priors']['per'], self.lc_data[n]['priors']['tmid'])
            self.lc_data[n]['time_upsample'] = np.linspace(min(self.lc_data[n]['time']), max(self.lc_data[n]['time']), 1000)
            self.lc_data[n]['phase_upsample'] = get_phase(self.lc_data[n]['time_upsample'], self.lc_data[n]['priors']['per'], self.lc_data[n]['priors']['tmid'])
            self.lc_data[n]['transit_upsample'] = transit(self.lc_data[n]['time_upsample'], self.lc_data[n]['priors'])

        # create an average value from all the local fits
        if rprs_in_local:
            try:
                self.parameters['rprs'] = np.asnumpy(np.mean(np.array(local_rprs, dtype=np.float64)))
                self.errors['rprs'] = np.asnumpy(np.std(np.array(local_rprs, dtype=np.float64)))
            except AttributeError:
                self.parameters['rprs'] = np.mean(local_rprs)
                self.errors['rprs'] = np.std(local_rprs)

        #import pdb; pdb.set_trace()

    def plot_bestfits(self):
        nrows = len(self.lc_data)//4+1
        # make sure there isn't an extra row
        if len(self.lc_data)%4 == 0:
            nrows -= 1

        fig,ax = plt.subplots(nrows, 4, figsize=(5+5*nrows, 5*nrows))

        # turn off all axes
        for i in range(nrows*4):
            ri = int(i/4)
            ci = i%4
            if ax.ndim == 1:
                ax[i].axis('off')
            else:
                ax[ri,ci].axis('off')

        # cycle the colors and markers
        markers = cycle(['o','v','^','<','>','s','*','h','H','D','d','P','X'])
        colors = cycle(['black','blue','green','orange','purple','grey','magenta','cyan','lime'])

        # plot observations
        for i in range(len(self.lc_data)):
            ri = int(i/4)
            ci = i%4
            ncolor = next(colors)
            nmarker = next(markers)

            model = transit(self.lc_data[i]['time'], self.lc_data[i]['priors'])
            airmass = np.exp(self.lc_data[i]['airmass']*self.lc_data[i]['priors']['a2'])
            detrend = self.lc_data[i]['flux']/(model*airmass)

            if ax.ndim == 1:
                ax[i].axis('on')
                ax[i].errorbar(self.lc_data[i]['time'], self.lc_data[i]['flux']/airmass/detrend.mean(), yerr=self.lc_data[i]['ferr']/airmass/detrend.mean(), 
                                ls='none', marker=nmarker, color=ncolor, alpha=0.5, zorder=1)
                
                ax[i].plot(self.lc_data[i]['time_upsample'], self.lc_data[i]['transit_upsample'], 'r-', zorder=2)
                ax[i].set_xlabel("Time [BJD]", fontsize=14)
                ax[i].set_ylabel("Relative Flux", fontsize=14)
                ax[i].set_title(f"{self.lc_data[i].get('name','')}", fontsize=16)
            else:
                ax[ri,ci].axis('on')
                ax[ri,ci].errorbar(self.lc_data[i]['time'], self.lc_data[i]['flux']/airmass/detrend.mean(), yerr=self.lc_data[i]['ferr']/airmass/detrend.mean(), 
                                   ls='none', marker=nmarker, color=ncolor, alpha=0.5, zorder=1)
                ax[ri,ci].plot(self.lc_data[i]['time_upsample'], self.lc_data[i]['transit_upsample'], 'r-', zorder=2)
                ax[ri,ci].set_xlabel("Time[BJD]", fontsize=14)
                ax[ri,ci].set_ylabel("Relative Flux", fontsize=14)
                ax[ri,ci].set_title(f"{self.lc_data[i].get('name','')}", fontsize=16)

        plt.tight_layout()
        return fig

    def plot_bestfit(self, title="", bin_dt=30./(60*24), alpha=0.05, ylim_sigma=5, phase_limits='median', show_legend=True, limit_legend=False, show_individual_fits=False):
        """
        Plot the best fit model and residuals

        Parameters
        ----------
        title : str
            Title for the plot

        bin_dt : float
            Bin size for plotting the residuals

        alpha : float
            Alpha value for plotting the data

        ylim_sigma : float
            Number of sigma to plot the residuals

        phase_limits : str
            'median' or 'all' to set the phase limits

        show_legend : bool
            Show the legend

        limit_legend : bool
            Limit the legend to 3 entries
        """
        f = plt.figure(figsize=(15,12))
        f.subplots_adjust(top=0.92,bottom=0.09,left=0.1,right=0.98, hspace=0)
        ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        axs[0].set_title(title, fontsize=18)
        axs[0].set_ylabel("Relative Flux", fontsize=14)
        axs[0].grid(True,ls='--')

        try:
            rprs2 = self.parameters['rprs']**2
            rprs2err = 2*self.parameters['rprs']*self.errors['rprs']
        except:
            rprs2 = self.lc_data[0]['priors']['rprs']**2
            rprs2err = 2*self.lc_data[0]['priors']['rprs']*self.lc_data[0]['errors']['rprs']

        lclabel1 = r"$R^{2}_{p}/R^{2}_{s}$ = %s $\pm$ %s" %(
            str(round_to_2(rprs2, rprs2err)),
            str(round_to_2(rprs2err))
        )
        
        lclabel2 = r"$T_{mid}$ = %s $\pm$ %s BJD$_{TDB}$" %(
            str(round_to_2(self.parameters['tmid'], self.errors.get('tmid',0))),
            str(round_to_2(self.errors.get('tmid',0)))
        )

        lclabel = lclabel1 + "\n" + lclabel2
        minp = 1
        maxp = 0

        min_std = 1
        # cycle the colors and markers
        markers = cycle(['o','v','^','<','>','s','*','h','H','D','d','P','X'])
        colors = cycle(['black','blue','green','orange','purple','grey','magenta','cyan','lime'])

        alldata = {
            'time': [],
            'flux': [],
            'detrend': [],
            'ferr': [],
            'residuals': [],
        }

        for n in range(len(self.lc_data)):
            ncolor = next(colors)
            nmarker = next(markers)
            alldata['time'].extend(self.lc_data[n]['time'].tolist())
            alldata['detrend'].extend(self.lc_data[n]['detrend'].tolist())
            alldata['flux'].extend(self.lc_data[n]['flux'].tolist())
            alldata['ferr'].extend(self.lc_data[n]['ferr'].tolist())
            alldata['residuals'].extend(self.lc_data[n]['residuals'].tolist())
            
            phase = get_phase(self.lc_data[n]['time'], self.parameters['per'], self.lc_data[n]['priors']['tmid'])
            try:
                si = np.argsort(np.array(phase))
            except AttributeError:
                si = np.argsort(phase)
            #bt2, br2, _ = time_bin(phase[si]*self.parameters['per'], self.lc_data[n]['residuals'][si]/np.median(self.lc_data[n]['flux'])*1e2, bin_dt)

            # plot data
            try:
                axs[0].errorbar(phase, self.lc_data[n]['detrend'], yerr=np.asnumpy(np.std(np.array(self.lc_data[n]['residuals']))/np.median(np.array(self.lc_data[n]['flux']))), 
                            ls='none', marker=nmarker, color=ncolor, zorder=1, alpha=alpha)
            except AttributeError:
                axs[0].errorbar(phase, self.lc_data[n]['detrend'], yerr=np.std(self.lc_data[n]['residuals'])/np.median(self.lc_data[n]['flux']), 
                            ls='none', marker=nmarker, color=ncolor, zorder=1, alpha=alpha)

            # plot residuals
            try:
                axs[1].plot(phase, self.lc_data[n]['residuals']/np.asnumpy(np.median(np.array(self.lc_data[n]['flux']))*1e2), color=ncolor, marker=nmarker, ls='none',
                         alpha=0.2)
            except AttributeError:
                axs[1].plot(phase, self.lc_data[n]['residuals']/np.median(self.lc_data[n]['flux'])*1e2, color=ncolor, marker=nmarker, ls='none',
                         alpha=0.2)

            # plot binned data
            try:
                bt2, bf2, bs = time_bin(np.array(phase[si])*np.array(self.lc_data[n]['priors']['per']), np.array(self.lc_data[n]['detrend'][si]), bin_dt)
            except AttributeError:
                bt2, bf2, bs = time_bin(phase[si]*self.lc_data[n]['priors']['per'], self.lc_data[n]['detrend'][si], bin_dt)

            if limit_legend:
                try:
                    axs[0].errorbar(np.asnumpy(bt2)/self.lc_data[n]['priors']['per'],np.asnumpy(bf2),yerr=np.asnumpy(bs),alpha=1,zorder=2,color=ncolor,ls='none',marker=nmarker)
                except AttributeError:
                    axs[0].errorbar(bt2/self.lc_data[n]['priors']['per'],bf2,yerr=bs,alpha=1,zorder=2,color=ncolor,ls='none',marker=nmarker)
            else:
                try:
                    axs[0].errorbar(np.asnumpy(bt2)/self.lc_data[n]['priors']['per'],np.asnumpy(bf2),yerr=np.asnumpy(bs),alpha=1,zorder=2,color=ncolor,ls='none',marker=nmarker,
                                label=r'{}: {:.2f} %'.format(self.lc_data[n].get('name',''),np.asnumpy(np.std(np.array(self.lc_data[n]['residuals'])/np.median(np.array(self.lc_data[n]['flux']))*1e2))))
                except AttributeError:
                    axs[0].errorbar(bt2/self.lc_data[n]['priors']['per'],bf2,yerr=bs,alpha=1,zorder=2,color=ncolor,ls='none',marker=nmarker,
                                label=r'{}: {:.2f} %'.format(self.lc_data[n].get('name',''),np.std(self.lc_data[n]['residuals']/np.median(self.lc_data[n]['flux'])*1e2)))

            # replace min and max for upsampled lc model
            minp = min(minp, min(phase))
            maxp = max(maxp, max(phase))
            try:
                min_std = min(min_std, np.asnumpy(np.std(np.array(self.lc_data[n]['residuals'])/np.median(np.array(self.lc_data[n]['flux'])))))
            except AttributeError:
                min_std = min(min_std, np.std(self.lc_data[n]['residuals']/np.median(self.lc_data[n]['flux'])))

            # plot individual best fit models
            if show_individual_fits:
                axs[0].plot(self.lc_data[n]['phase_upsample'], self.lc_data[n]['transit_upsample'], color=ncolor, zorder=3, alpha=0.5)

        # create binned plot for all the data
        for k in alldata.keys():
            try:
                alldata[k] = np.asnumpy(np.array(alldata[k]))
            except AttributeError:
                alldata[k] = np.array(alldata[k])
            
        phase = get_phase(alldata['time'], self.parameters['per'], self.lc_data[n]['priors']['tmid'])
        try:
            si = np.argsort(np.array(phase))
            bt, br, _ = time_bin(np.aRRAY(phase[si])*np.array(self.parameters['per']), np.array(alldata['residuals'][si])/np.median(np.array(alldata)['flux']), 2*bin_dt)
            bt, bf, bs = time_bin(phase[si]*self.parameters['per'], alldata['detrend'][si], 2*bin_dt)
        except AttributeError:
            si = np.argsort(phase)
            bt, br, _ = time_bin(phase[si]*self.parameters['per'], alldata['residuals'][si]/np.median(alldata['flux']), 2*bin_dt)
            bt, bf, bs = time_bin(phase[si]*self.parameters['per'], alldata['detrend'][si], 2*bin_dt)

        try:
            axs[0].errorbar(bt/self.parameters['per'],bf,yerr=bs,alpha=1,zorder=2,color='white',ls='none',marker='o',ms=15,
                        markeredgecolor='black',
                        ecolor='black',
                        label=r'Binned Data: {:.2f} %'.format(np.asnumpy(np.std(br)*1e2)))
        except AttributeError:
            axs[0].errorbar(bt/self.parameters['per'],bf,yerr=bs,alpha=1,zorder=2,color='white',ls='none',marker='o',ms=15,
                        markeredgecolor='black',
                        ecolor='black',
                        label=r'Binned Data: {:.2f} %'.format(np.std(br)*1e2))

        axs[1].plot(bt/self.parameters['per'],br*1e2,color='white',ls='none',marker='o',ms=11,markeredgecolor='black')
        
        # best fit model
        try:
            self.time_upsample = np.asnumpy(np.linspace(minp*np.array(self.parameters['per'])+np.array(self.parameters['tmid']), 
                                         maxp*np.array(self.parameters['per'])+np.array(self.parameters['tmid']), 10000))
        except AttributeError:
            self.time_upsample = np.linspace(minp*self.parameters['per']+self.parameters['tmid'], 
                                         maxp*self.parameters['per']+self.parameters['tmid'], 10000)

        self.transit_upsample = transit(self.time_upsample, self.parameters)
        self.phase_upsample = get_phase(self.time_upsample, self.parameters['per'], self.parameters['tmid'])
        try:
            sii = np.argsort(np.array(self.phase_upsample))
        except AttributeError:
            sii = np.argsort(self.phase_upsample)
        axs[0].plot(self.phase_upsample[sii], self.transit_upsample[sii], 'r-', zorder=3, label=lclabel, lw=3)

        # set up axes limits
        axs[0].set_xlim([min(self.phase_upsample), max(self.phase_upsample)])
        axs[0].set_xlabel("Phase ", fontsize=14)
        axs[0].set_ylim([1-self.parameters['rprs']**2-ylim_sigma*min_std, 1+ylim_sigma*min_std])
        axs[1].set_xlim([min(self.phase_upsample), max(self.phase_upsample)])
        axs[1].set_xlabel("Phase", fontsize=14)
        axs[1].set_ylim([-5*min_std*1e2, 5*min_std*1e2])

        # compute average min and max for all the data
        mins = []; maxs = []
        for n in range(len(self.lc_data)):
            mins.append(min(self.lc_data[n]['phase']))
            maxs.append(max(self.lc_data[n]['phase']))

        # set up phase limits
        if isinstance(phase_limits, str):
            if phase_limits == "minmax":
                axs[0].set_xlim([min(self.phase_upsample), max(self.phase_upsample)])
                axs[1].set_xlim([min(self.phase_upsample), max(self.phase_upsample)])
            elif phase_limits == "median":
                try:
                    axs[0].set_xlim([np.asnumpy(np.median(mins)), np.asnumpy(np.median(maxs))])
                    axs[1].set_xlim([np.asnumpy(np.median(mins)), np.asnumpy(np.median(maxs))])
                except AttributeError:
                    axs[0].set_xlim([np.median(np.array(mins)), np.median(np.array(maxs))])
                    axs[1].set_xlim([np.median(np.array(mins)), np.median(np.array(maxs))])
            else:
                axs[0].set_xlim([min(self.phase_upsample), max(self.phase_upsample)])
                axs[1].set_xlim([min(self.phase_upsample), max(self.phase_upsample)])
        elif isinstance(phase_limits, list):
            axs[0].set_xlim([phase_limits[0], phase_limits[1]])
            axs[1].set_xlim([phase_limits[0], phase_limits[1]])
        elif isinstance(phase_limits, tuple):
            axs[0].set_xlim([phase_limits[0], phase_limits[1]])
            axs[1].set_xlim([phase_limits[0], phase_limits[1]])
        else:
            axs[0].set_xlim([min(self.phase_upsample), max(self.phase_upsample)])
            axs[1].set_xlim([min(self.phase_upsample), max(self.phase_upsample)])

        axs[0].get_xaxis().set_visible(False)
        axs[1].set_ylabel("Residuals [%]", fontsize=14)
        axs[1].grid(True,ls='--',axis='y')
    
        if show_legend:
            axs[0].legend(loc='best',ncol=len(self.lc_data)//7+1)
    
        return f,axs

    def plot_stack(self, title="", bin_dt=30./(60*24), dy=0.02):
        f, ax = plt.subplots(1,figsize=(9,12))
        
        ax.set_title(title)
        ax.set_ylabel("Relative Flux", fontsize=14)
        ax.grid(True,ls='--')

        rprs2 = self.parameters['rprs']**2
        rprs2err = 2*self.parameters['rprs']*self.errors['rprs']
        lclabel1 = r"$R^{2}_{p}/R^{2}_{s}$ = %s $\pm$ %s" %(
            str(round_to_2(rprs2, rprs2err)),
            str(round_to_2(rprs2err))
        )
        
        lclabel2 = r"$T_{mid}$ = %s $\pm$ %s BJD$_{TDB}$" %(
            str(round_to_2(self.parameters['tmid'], self.errors.get('tmid',0))),
            str(round_to_2(self.errors.get('tmid',0)))
        )

        lclabel = lclabel1 + "\n" + lclabel2
        minp = 1
        maxp = 0

        min_std = 1
        # cycle the colors and markers
        markers = cycle(['o','v','^','<','>','s','*','h','H','D','d','P','X'])
        colors = cycle(['black','blue','green','orange','purple','grey','magenta','cyan','lime'])
        for n in range(len(self.lc_data)):
            ncolor = next(colors)
            nmarker = next(markers)

            phase = get_phase(self.lc_data[n]['time'], self.parameters['per'], self.lc_data[n]['priors']['tmid'])
            try:
                si = np.argsort(np.array(phase))
                bt2, br2, _ = time_bin(phase[si]*self.parameters['per'], self.lc_data[n]['residuals'][si]/np.asnumpy(np.median(np.array(self.lc_data[n]['flux'])))*1e2, bin_dt)
            except AttributeError:
                si = np.argsort(phase)
                bt2, br2, _ = time_bin(phase[si]*self.parameters['per'], self.lc_data[n]['residuals'][si]/np.median(self.lc_data[n]['flux'])*1e2, bin_dt)
            
            # plot data
            try:
                ax.errorbar(phase, self.lc_data[n]['detrend']-n*dy, yerr=np.asnumpy(np.std(np.array(self.lc_data[n]['residuals']))/np.median(np.array(self.lc_data[n]['flux']))), 
                            ls='none', marker=nmarker, color=ncolor, zorder=1, alpha=0.25)
            except AttributeError:
                ax.errorbar(phase, self.lc_data[n]['detrend']-n*dy, yerr=np.std(self.lc_data[n]['residuals'])/np.median(self.lc_data[n]['flux']), 
                            ls='none', marker=nmarker, color=ncolor, zorder=1, alpha=0.25)

            # plot binned data
            try:
                bt2, bf2, bs = time_bin(np.array(phase[si])*np.array(self.lc_data[n]['priors']['per']), np.array(self.lc_data[n]['detrend'][si])-n*dy, bin_dt)
                ax.errorbar(np.asnumpy(bt2)/self.lc_data[n]['priors']['per'],np.asnumpy(bf2),yerr=np.asnumpy(bs),alpha=1,zorder=2,color=ncolor,ls='none',marker=nmarker)
            except AttributeError:
                bt2, bf2, bs = time_bin(phase[si]*self.lc_data[n]['priors']['per'], self.lc_data[n]['detrend'][si]-n*dy, bin_dt)
                ax.errorbar(bt2/self.lc_data[n]['priors']['per'],bf2,yerr=bs,alpha=1,zorder=2,color=ncolor,ls='none',marker=nmarker)

            # replace min and max for upsampled lc model
            minp = min(minp, min(phase))
            maxp = max(maxp, max(phase))
            try:
                min_std = min(min_std, np.asnumpy(np.std(np.array(self.lc_data[n]['residuals'])/np.median(np.array(self.lc_data[n]['flux'])))))
            except AttributeError:
                min_std = min(min_std, np.std(self.lc_data[n]['residuals']/np.median(self.lc_data[n]['flux'])))

            # best fit model
            try:
                self.time_upsample = np.asnumpy(np.linspace(minp*np.array(self.parameters['per'])+np.array(self.parameters['tmid']), 
                                            maxp*np.array(self.parameters['per'])+np.array(self.parameters['tmid']), 10000))
            except AttributeError:   
                self.time_upsample = np.linspace(minp*self.parameters['per']+self.parameters['tmid'], 
                                            maxp*self.parameters['per']+self.parameters['tmid'], 10000)
            self.transit_upsample = transit(self.time_upsample, self.parameters)
            self.phase_upsample = get_phase(self.time_upsample, self.parameters['per'], self.parameters['tmid'])
            try:
                sii = np.argsort(np.array(self.phase_upsample))
            except AttributeError:  
                sii = np.argsort(self.phase_upsample)
            ax.plot(self.phase_upsample[sii], self.transit_upsample[sii]-n*dy, ls='-', color=ncolor, zorder=3, label=self.lc_data[n].get('name',''))

        ax.set_xlim([min(self.phase_upsample), max(self.phase_upsample)])
        ax.set_xlabel("Phase ", fontsize=14)
        ax.set_ylim([1-self.parameters['rprs']**2-5*min_std-n*dy, 1+5*min_std])
        ax.get_xaxis().set_visible(False)
        ax.legend(loc='best')
        return f,ax


if __name__ == "__main__":

    prior = {
        'rprs': 0.02,  # Rp/Rs
        'ars': 14.25,  # a/Rs
        'per': 3.33,  # Period [day]
        'inc': 88.5,  # Inclination [deg]
        'u0': 0, 'u1': 0, 'u2': 0, 'u3': 0,  # limb darkening (nonlinear)
        'ecc': 0.5,  # Eccentricity
        'omega': 120,  # Arg of periastron
        'tmid': 0.75,  # Time of mid transit [day],
        'a1': 50,  # Airmass coefficients
        'a2': 0.,  # trend = a1 * np.exp(a2 * airmass)

        'teff': 5000,
        'tefferr': 50,
        'met': 0,
        'meterr': 0,
        'logg': 3.89,
        'loggerr': 0.01
    }

    # example generating LD coefficients
    from pylightcurve import exotethys

    u0, u1, u2, u3 = exotethys(prior['logg'], prior['teff'], prior['met'], 'TESS', method='claret',
                               stellar_model='phoenix')

    prior['u0'], prior['u1'], prior['u2'], prior['u3'] = u0, u1, u2, u3

    time = np.linspace(0.7, 0.8, 1000)  # [day]

    # simulate extinction from airmass
    stime = time - time[0]
    alt = 90 * np.cos(4 * stime - np.pi / 6)
    # airmass = 1./np.cos(np.deg2rad(90-alt))
    airmass = np.zeros(time.shape[0])

    # GENERATE NOISY DATA
    data = transit(time, prior) * prior['a1'] * np.exp(prior['a2'] * airmass)
    data += np.random.normal(0, prior['a1'] * 250e-6, len(time))
    dataerr = np.random.normal(300e-6, 50e-6, len(time)) + np.random.normal(300e-6, 50e-6, len(time))

    # add bounds for free parameters only
    mybounds = {
        'rprs': [0, 0.1],
        'tmid': [prior['tmid'] - 0.01, prior['tmid'] + 0.01],
        'ars': [13, 15],
        # 'a2': [0, 0.3] # uncomment if you want to fit for airmass
        # never list 'a1' in bounds, it is perfectly correlated to exp(a2*airmass)
        # and is solved for during the fit
    }

    myfit = lc_fitter(time, data, dataerr, airmass, prior, mybounds, mode='ns')

    for k in myfit.bounds.keys():
        print(f"{myfit.parameters[k]:.6f} +- {myfit.errors[k]}")

    fig, axs = myfit.plot_bestfit()
    plt.tight_layout()
    plt.show()

    fig = myfit.plot_triangle()
    plt.tight_layout()
    plt.show()
