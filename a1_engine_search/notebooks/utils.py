from numpy import dot
from numpy.linalg import norm
from scipy import spatial

def cos_sim_np(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def cos_sim_scipy(a, b):
    cos_sim = 1 - spatial.distance.cosine(a, b)  #distance = 1 - similarlity, because scipy only gives distance
    return cos_sim
