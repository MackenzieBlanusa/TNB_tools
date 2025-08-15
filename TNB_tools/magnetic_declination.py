import numpy as np

def calculate_direction(east,north):
    """
    This function calculates direction given east and north velocity components 

    ------
    Parameters 
    ------
    Inputs:
        - east: east velocity component; array 
        - north: north velocity component; array

    Outputs:
        - direction: calculated direction [degrees]; array 
    """
    
    direction = np.mod(np.arctan2(east,north)*180/np.pi,360)

    return direction 
    

def calculate_true_direction(direction,magnetic_declination):
    """
    Function that calculates true direction given a direction array in degrees and magnetic declination
    in degrees 

    -----
    Parameters
    -----

    Inputs: 
        - direction: array; direction in degrees 
        - magnetic_declination: float; magnetic declination angle in degrees 

    Outputs: 
        - true_direction: array; true direction in degrees (i.e. direction accounted for magnetic 
        declination) 
    
    """

    true_direction = np.mod(direction+magnetic_declination,360)

    return true_direction 
    
def rotate_velocity(u,v,angle):
    """Function that rotates velocity component vectors using rotation equations
       
       Inputs: v: u velocity, v velocity, angle: angle between vectors [radians]
       Outputs: rotated v velocity vectors 
    """
    u_prime = u*np.cos(angle) - v*np.sin(angle)
    v_prime = u*np.sin(angle) + v*np.cos(angle)
    
    return u_prime,v_prime
