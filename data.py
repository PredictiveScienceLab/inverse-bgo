"""
Some utilities related to the data.

Author:
    Ilias Bilionis

Date:
    7/19/2015

"""


__all__ = ['load_catalysis_data']


import numpy as np 


def load_catalysis_data():                                                         
    """                                                                            
    Loads the catalysis data and casts them to the right format.                   
    """                                                                            
    data = np.loadtxt('catalysis_data.txt')
    return data.flatten() / 500.