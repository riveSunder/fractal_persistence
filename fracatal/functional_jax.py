from jax import numpy as np
import numpy.random as npr


def ft_convolve(grid, kernel):                                                  
                                             
    if np.shape(kernel) != np.shape(grid):                                     

        diff_h  = np.shape(grid)[-2] - np.shape(kernel)[-2] 
        diff_w =  np.shape(grid)[-1] - np.shape(kernel)[-1] 
        pad_h = diff_h // 2
        pad_w = diff_w // 2

        rh, rw = diff_h % pad_h, diff_w % pad_w

        if rh:
            hp = rh
            hm = 0
        else:
            hp = 1
            hm = -1

        if rw:
            wp = rw
            wm = 0
        else:
            wp = 1
            wm = -1

        padded_kernel = np.pad(kernel, \
                ((0,0), (0,0), (pad_h+hp, pad_h+hm), (pad_w+wp, pad_w+wm)))

    else:                                                                       
        padded_kernel = kernel                                                  
                                                                                
    fourier_kernel = np.fft.fft2(np.fft.fftshift(padded_kernel, axes=(-2,-1)), axes=(-2,-1))
    fourier_grid = np.fft.fft2(np.fft.fftshift(grid, axes=(-2,-1)), axes=(-2,-1))
    fourier_product = fourier_grid * fourier_kernel 
    real_spatial_convolved = np.real(np.fft.ifft2(fourier_product, axes=(-2,-1)))
    convolved = np.fft.ifftshift(real_spatial_convolved, axes=(-2, -1))
                                                                                
    return convolved 

def make_gaussian(a, m, s):

    eps = 1e-9

    def gaussian(x):

        return a * np.exp(-((x-m)/(eps+s))**2 / 2)

    return gaussian

def make_mixed_gaussian(amplitudes, means, std_devs):

    def gaussian_mixture(x):
        results = 0.0 * x
        eps = 1e-9 # prevent division by 0

        for a, m, s in zip(amplitudes, means, std_devs):
            my_gaussian = make_gaussian(a, m, s)
            results += my_gaussian(x)

        return results

    return gaussian_mixture

def make_kernel_field(kernel_radius, dim=126):

    #dim = kernel_radius * 2 + 1


    x =  np.arange(-dim / 2, dim / 2 + 1, 1)
    xx, yy = np.meshgrid(x,x)

    rr = np.sqrt(xx**2 + yy**2) / kernel_radius

    return rr

def make_update_function(mean, standard_deviation):

    my_gaussian = make_gaussian(1.0, mean, standard_deviation)

    def lenia_update(x):
        """
        lenia update
        """
        return 2 * my_gaussian(x) - 1

    return lenia_update

def make_update_step(update_function, kernel, dt, clipping_function = lambda x: x, decimals=None):

    if decimals is not None:
        r = lambda x: np.round(x, decimals=decimals)
    else:
        r = lambda x: x

    def update_step(grid):


        neighborhoods = r(ft_convolve(r(grid), r(kernel)))
        dgrid_dt = r(update_function(neighborhoods))

        new_grid = r(clipping_function(r(grid) + dt * dgrid_dt))

        return new_grid

    return update_step

def make_make_kernel_function(amplitudes, means, standard_deviations, dim=126):

    def make_kernel(kernel_radius):

        gm = make_mixed_gaussian(amplitudes, means, standard_deviations)
        rr = make_kernel_field(kernel_radius, dim=dim)
        kernel = gm(rr)[None,None,:,:]
        kernel = kernel / kernel.sum()

        return kernel

    return make_kernel

# smooth life
def sigmoid_1(x, mu, alpha, gamma=1):
    return 1 / (1 + np.exp(-4 * (x - mu) / alpha))

def get_smooth_steps_fn(intervals, alpha=0.0125):
    """
    construct an update function from intervals.
    input intervals is a list of lists of interval bounds,
    each element contains the start and end of an interval.

    # this code was adopated from rivesunder/yuca
    """

    def smooth_steps_fn(x):

        result = np.zeros_like(x)
        for bounds in intervals:
            result += sigmoid_1(x, bounds[0], alpha) * (1 - sigmoid_1(x, bounds[1], alpha))

        return result

    return smooth_steps_fn

def make_make_smoothlife_kernel_function(r_inner, r_outer, dim=126):
    """
    r_inner and r_outer are 1/3. and 1.05 for Rafler's SmoothLife glider configuration
    """

    def make_smoothlife_kernel(kernel_radius=10):

        rr = make_kernel_field(kernel_radius, dim=dim)
        kernel = np.ones_like(rr)
        kernel = kernel.at[rr < r_inner].set(0.0)
        kernel = kernel.at[rr >= r_outer].set(0.0)

        kernel = (kernel / kernel.sum())[None,None,:,:]

        return kernel

    return make_smoothlife_kernel

def make_smooth_interval(alpha=0.1470, intervals=[[0.2780, 0.3650]]):

    def smooth_interval(x):

        result = 0.0 * x

        for bounds in intervals:
            result += sigmoid_1(x, bounds[0], alpha) * (1-sigmoid_1(x, bounds[1], alpha))

        return 2 * result - 1

    return smooth_interval

def make_smoothlife_update_function(intervals=[[0.2780, 0.3650]], \
                                    alpha=0.028):

    smooth = make_smooth_interval(alpha=alpha, intervals=intervals)

    def smoothlife_update(x):
        """
        smoothlife half-update
        """
        return 2 * smooth(x) - 1

    return smoothlife_update

def make_smoothlife_update_step(genesis_function, persistence_function, \
                                kernel, inner_kernel, dt, clipping_function = lambda x: x, \
                                decimals=None):

    if decimals is not None:
        r = lambda x: np.round(x, decimals=decimals)
    else:
        r = lambda x: x

    def update_step(grid):

        neighborhoods = r(ft_convolve(r(grid), r(kernel)))
        inner = r(ft_convolve(r(grid), r(inner_kernel)))

        genesis = r(genesis_function(neighborhoods))
        persistence = r(persistence_function(neighborhoods))
        dgrid_dt = (1-inner) * genesis + inner * persistence

        new_grid = r(clipping_function(r(grid) + dt * dgrid_dt))

        return new_grid

    return update_step

