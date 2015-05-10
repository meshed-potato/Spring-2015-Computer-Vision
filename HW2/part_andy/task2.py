import numpy as np

IMG_FILE = "HW2_image.txt"
WORLD_FILE = "HW2_world.txt"
NUM_OF_POINTS = 10

def readpoints(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data += line.strip(' ').strip('\n').split('  ')
    return data

def construct_world_points():
    world_points = readpoints(WORLD_FILE)
    x_world = np.zeros((NUM_OF_POINTS,4)) 
    for i in range(NUM_OF_POINTS):
        for j in range(3):
            x_world[i,j] = world_points[j*NUM_OF_POINTS+i]
        x_world[i,3] = 1  
    return x_world

def construct_img_points():
    image_points = readpoints(IMG_FILE)
    x_img = np.zeros((NUM_OF_POINTS,3))
    for i in range(NUM_OF_POINTS):
        for j in range(2):
            x_img[i,j] = image_points[j*NUM_OF_POINTS+i]
        x_img[i,2] = 1
    return x_img

def construct_a(x_world,x_img):
    A = []
    for i in range(NUM_OF_POINTS):
        temp1 = [0,0,0,0]
        temp1 += [-x_img[i][2] * p for p in x_world[i]]
        temp1 += [x_img[i][1] * p for p in x_world[i]]

        temp2 = [x_img[i][2] * p for p in  x_world[i]]
        temp2 += [0,0,0,0] 
        temp2 += [-x_img[i][0] * p for p in  x_world[i]]
        A.append(temp1)
        A.append(temp2)
    A = np.array(A)
    #print '\nA:\n',A
    return A



def verify(x_world,x_img,p):
    reproj_res = np.zeros((10,3))
    print '\n\n'
    for i in range(NUM_OF_POINTS):
        x = np.array(x_world[i])
        reproj_x = np.dot(p,x)
        reproj_res[i,:] = reproj_x/reproj_x[2]
    print 'reproj_res,\n',reproj_res
    print 'x_img\n',x_img
    diff = x_img - reproj_res
    print 'diff\n',diff

def get_svd_sol(A):
    U, s, V = np.linalg.svd(A, full_matrices=True)
    i_min = np.argmin(s) #i_min is 11 since s is decreasing
    min_eigenvector = V[i_min,:]

    return min_eigenvector





if __name__ == "__main__":

    """q1"""
    #sort points
    x_world = construct_world_points()
    x_img = construct_img_points()
    print 'x_world:\n',x_world,
    print '\nx_img:\n',x_img

    #Construct A
    A = construct_a(x_world,x_img)

    #compute SVD for A
    p = get_svd_sol(A)
    p = p.reshape((3,4))
    print 'p:\n',p

    #verity
    verify(x_world,x_img,p)

    """q2"""
    camera_coor_homo = get_svd_sol(p) #get the homogeneous solution
    camera_coor_homo = camera_coor_homo/camera_coor_homo[camera_coor_homo.size-1]
    camera_coor = camera_coor_homo[:3]
    print '\nthe camera coordinate is:\n',camera_coor
