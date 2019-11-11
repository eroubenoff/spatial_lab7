"""
Lab 7

Ethan Roubenoff

To be loaded into jupyter notebook, eventuall
"""

import random
random.seed(0)
import numpy as np
np.random.seed(0)
import requests as re
from skimage import io
import matplotlib.pyplot as plt


def download_file(url, local_filename):
    r = re.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size = 1024):
            if chunk:
                f.write(chunk)
    return(local_filename)

def q1():

    my_api_key = "YKZXKO7aW8M9UgLfxoLMlPb3W9TO7mB6rBN0gK7k"

    Berkeley_lat = "37.87"
    Berkeley_lon = "-122.26"
    date = "2014-05-21"

    base_url = "https://api.nasa.gov/planetary/earth/imagery/"
    image_query = base_url+"?lat="+Berkeley_lat+"&lon="+Berkeley_lon+"&date="+date
    image_query = image_query+"&api_key="+my_api_key
    img_a = image_query

    r = re.get(image_query)
    img_b = r.url

    r = re.get(base_url, params = 
            {"lat" : Berkeley_lat, "lon" : Berkeley_lon, 
                "date": date, "api_key":my_api_key})
    img_c = r.url
    assert img_a == img_b == img_c

    # Question 1
    nasa_reply_url = r.json()["url"]
    download_file(nasa_reply_url, "img.png")
    im = io.imread("img.png")
    plt.figure()
    plt.imshow(im)
    plt.show(block=False)

    print("Numpy array im is "+ str(im.shape))

    # Question 2
    fig, axes = plt.subplots(ncols = 3)
    axes[0].imshow(im[:, :, 0], cmap='Greys')
    axes[0].set_title("Red")
    axes[1].imshow(im[:, :, 1], cmap='Greys')
    axes[1].set_title("Green")
    axes[2].imshow(im[:, :, 2], cmap='Greys')
    axes[2].set_title("Blue")
    plt.show(block=False)

    # Question 3
    Berkeley_lat_S1 = str(37.87 - 0.025) # One block South
    Berkeley_lat_N1 = str(37.87 + 0.025) # One block North
    Berkeley_lat_N2 = str(37.87 + 0.050) # Nwo blocks North

    # Download the three files
    S1 = re.get(base_url, params = 
            {"lat" : Berkeley_lat_S1, "lon" : Berkeley_lon, 
                "date": date, "api_key":my_api_key})
    download_file(S1.json()['url'], "S1.png")
    S1 = io.imread("S1.png")
    N1 = re.get(base_url, params = 
            {"lat" : Berkeley_lat_N1, "lon" : Berkeley_lon, 
                "date": date, "api_key":my_api_key})
    download_file(N1.json()['url'], "N1.png")
    N1 = io.imread("N1.png")
    N2 = re.get(base_url, params = 
            {"lat" : Berkeley_lat_N2, "lon" : Berkeley_lon, 
                "date": date, "api_key":my_api_key})
    download_file(N2.json()['url'], "N2.png")
    N2 = io.imread("N2.png")
    
    # Stack images together
    stack  = np.vstack((N2, N1, im, S1))
    plt.figure()
    plt.imshow(stack)
    plt.show(block=False)

    # Question 4
    im = stack
    fig, axes = plt.subplots(ncols = 2, nrows = 2)
    axes[0,0].scatter(im[:,:,0].flatten(), im[:,:,1].flatten(),alpha=0.1,s=1)
    axes[0,0].set_title("Red vs. Green")
    axes[0,1].scatter(im[:,:,2].flatten(), im[:,:,1].flatten(),alpha=0.1,s=1)
    axes[0,1].set_title("Blue vs. Green")
    axes[1,0].scatter(im[:,:,0].flatten() > 120, im[:,:,1].flatten() > 120,alpha=0.1,s=1)
    axes[1,0].set_title("Red vs. Green Adjusted")
    axes[1,1].scatter(im[:,:,2].flatten() > 120, im[:,:,1].flatten() > 120,alpha=0.1,s=1)
    axes[1,1].set_title("Blue vs. Green Adjusted")
    plt.show(block=False)

    i_ind = im.shape[0]-1
    j_ind = im.shape[1]-1
    k_ind = im.shape[2]-1

    mask = np.zeros((i_ind, j_ind))

    for i in range(0, i_ind):
        for j in range(0, j_ind):
            for k in range(0, k_ind):
                if im[i, j, 0] < 120 and im[i, j, 1] < 120 and im[i, j, 2] < 120: 
                    mask[i, j] = 1 

    print(np.sum(mask))

    plt.figure()
    plt.imshow(mask, cmap="Greys")
    plt.show()

def init_matrix(x_dim = 10, y_dim = 10): 
    """ Initializes a 10x10 km matrix with a single observation.
    The return matrix is a 10x10 binary matrix.
    """
    ret = np.zeros((x_dim, y_dim))
    x_rand = np.random.randint(0, x_dim - 1)
    y_rand = np.random.randint(0, y_dim - 1)
    ret[x_rand, y_rand] = 1

    return(ret)

def matrix_step(loc_list): 
    """ Takes a list of coordinates and advances the snow leoppard by
    one day.  Each day has a 1/8 prob of entering an adjacent cell.
    If in corner, 1/3 prob of adjacent cell, etc. 
    """
    prev_loc = loc_list
    x_min = 0 if prev_loc[0] - 1 < 0  else prev_loc[0] - 1
    x_max = 10 if prev_loc[0] + 1 > 10 else prev_loc[0] + 1
    y_min = 0 if prev_loc[1] - 1 < 0  else prev_loc[1] - 1
    y_max = 10 if prev_loc[1] + 1 > 10 else prev_loc[1] + 1

    next_loc = prev_loc 
              
    while(prev_loc == next_loc):
        next_loc = (np.random.randint(x_min, x_max + 1),
                    np.random.randint(y_min, y_max + 1))

    return(next_loc)

def plot_matrix(loc_list):
    """ Plots the path of a leoppard taking a matrix and a location list
    No return but prints the path
    """
    x_list = [x[0] for x in loc_list]
    y_list = [y[1] for y in loc_list]

    # print(x_list, y_list)
    # plt.figure()

    plt.plot(x_list, y_list)


def run_sims(nsims = 10, plot = True):
    """ Initializes a matrix with ncats snow leoppards and 
    advances them  nsims times.

    mat takes value 1 if the leoppard is currently there
    mat takes value 0 if the leoppard has never been there
    mat takes value -n if it was there n times ago
    """
    mat = init_matrix()
    init_loc = np.where(mat == 1)
    init_loc = (init_loc[0][0], init_loc[1][0])
    loc_list = [init_loc]

    for _ in range(nsims):
        loc_list.append(matrix_step(loc_list[-1])) # the most recent entry in the list
        # print(loc_list[-2], loc_list[-1])

    if plot:
        plot_matrix(loc_list)
    return(loc_list)

def run_model(nsims = 10, ncats = 1):
    """ Runs the run_sims model for each cat
    Retuns a 10x10xNcats matrx
    """
    loc_lists = []
    plt.figure()
    for i in range(ncats):
        loc_lists.append(run_sims(nsims = nsims))
        plot_matrix(loc_lists[-1])
    return(loc_lists)

def camera(nsims, ncats, loc_lists):
    camera_list = []
    for cat in range(ncats):
        for sim in range(nsims):
            if loc_lists[cat][sim] == (3, 3):
                camera_list.append(sim)
    return(camera_list)
 
def mcmc(ncats = 1, nsims = 100, nreps = 1000): 
    ret = []
    for i in range(nreps):
        l = run_model(nsims = nsims, ncats = ncats)
        c = camera(nsims = 100, ncats = 1, loc_lists = l)
        ret.append(len(c))

    return(ret)

if __name__ == "__main__":
     # q1()
     # q2

    
     #plt.figure()
     #run_sims(nsims = 10)
     #plt.figure()
     #run_sims(nsims = 100)
     #run_model(nsims = 10, ncats = 1)
     #run_model(nsims = 10, ncats = 2)
     #run_model(nsims = 100, ncats = 10)
    
     l = run_model(nsims = 100, ncats = 3)
     c = camera(nsims = 100, ncats = 3, loc_lists = l)
     plt.plot(3, 3, "b*")
     plt.figure()
     plt.hist(c)
     r = mcmc()
     print(r)
     # plt.figure()
     # plt.hist(r)
     plt.show()

