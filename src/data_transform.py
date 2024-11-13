import numpy as np
def custom_transform(data):
    """
    Transform the `spiral.csv` data such that it can be more easily classified.

    To pass test_custom_transform_hard, your transformation should create at
    most three features and should allow a LogisticRegression model to achieve
    at least 90% accuracy.

    You can use free_response.q2.visualize_spiral() to visualize the spiral
    as we give it to you, and free_response.q2.visualize_transform() to
    visualize the 3D data transformation you implement here.

    Args:
        data: a Nx2 matrix from the `spiral.csv` dataset.

    Returns:
        A transformed data matrix that is (more) easily classified.
    """
    #K Lo K

    #run cos on x, y, then find distance from

    angle = np.arctan(data[:,1]/data[:,0]).reshape((data.shape[0],1))


    val_x_copy = np.sign(np.cos((data[:,0]/np.cos(angle).T))).reshape((data.shape[0],1))
    val_y_copy = np.sign((data[:,1]/np.sin(angle).T)).reshape((data.shape[0],1))
    val_x = np.where(val_y_copy < 0, -1*val_x_copy, val_x_copy)
    val_y = np.where(val_x_copy < 0, -1*val_y_copy, val_y_copy)





    
    res = np.concatenate((val_x,val_y_copy),axis =1)



    
    return res
