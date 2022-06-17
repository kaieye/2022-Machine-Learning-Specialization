from lab_utils_common import *
from ipywidgets import Output
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib.patches import FancyArrowPatch
import time

def map_one_feature(X1, degree):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    out = []
    str = ""
    k = 0
    for i in range(1, degree+1):
        out.append((X1**i))
        str = str + f"w_{{{k}}}{munge('x_0',i)} + "
        k += 1
    str = str + ' b' #add b to text equation, not to data
    return np.stack(out, axis=1), str 


def map_feature(X1, X2, degree):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)

    out = []
    str = ""
    k = 0
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
            str = str + f"w_{{{k}}}{munge('x_0',i-j)}{munge('x_1',j)} + "
            k += 1
    #print(str + 'b')
    return np.stack(out, axis=1), str+' b'

def munge(base,exp):
    if exp == 0:
        return ('')
    elif exp == 1:
        return (base)
    else:
        return (base + f'^{{{exp}}}')

def plot_decision_boundary(ax, x0r,x1r, y, predict,  w, b, scaler = False, mu=None, sigma=None, degree=None):
    """
    Plots a decision boundary 
     Args:
      x0r : (array_like Shape (1,1)) range (min, max) of x0
      x1r : (array_like Shape (1,1)) range (min, max) of x1
      y   : (array_like Shape (m, )) target values of y
      predict : function to predict z values    
      scalar : (boolean) scale data or not
    """

    h = .01  # step size in the mesh
    # create a mesh to plot in
    xx, yy = np.meshgrid(np.arange(x0r[0], x0r[1], h),
                         np.arange(x0r[0], x0r[1], h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    points = np.c_[xx.ravel(), yy.ravel()]
    Xm,_ = map_feature(points[:, 0], points[:, 1],degree)
    if scaler:
        Xm = (Xm - mu)/sigma
    Z = predict(Xm, w, b)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    contour = ax.contour(xx, yy, Z, levels = [0.5], colors='g')
    return(contour)


def plot_decision_boundary_sklearn(x0r,x1r, y,predict,  scaler = False):
    """
    Plots a decision boundary 
     Args:
      x0r : (array_like Shape (1,1)) range (min, max) of x0
      x1r : (array_like Shape (1,1)) range (min, max) of x1
      y   : (array_like Shape (m, )) target values of y
      predict : function to predict z values    
    """

    h = .01  # step size in the mesh
    # create a mesh to plot in
    xx, yy = np.meshgrid(np.arange(x0r[0], x0r[1], h),
                         np.arange(x0r[0], x0r[1], h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    points = np.c_[xx.ravel(), yy.ravel()]
    Xm = map_feature(points[:, 0], points[:, 1],degree)
    if scaler:
        Xm = scaler.transform(Xm)
    Z = predict(Xm)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='g') 
    #plot_data(X_train,y_train)


# for debug
#output = Output() # sends hidden error messages to display when using widgets
#display(output)

class plt_overfit():

#    @output.capture()  # debug
    def __init__(self, X, y, w_in, b_in):
        fig = plt.figure( figsize=(10,8))
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.set_facecolor('#ffffff') #white
        gs  = GridSpec(5, 3, figure=fig)
        ax0 = fig.add_subplot(gs[0:3, :])
        ax1 = fig.add_subplot(gs[4, :])
        ax2 = fig.add_subplot(gs[5, :])
        self.ax = [ax0,ax1,ax2]

        plot_data(X_train, y_train, ax0, s=10, loc='lower right')
        ax0.set_title("Logistic data set with noise")
        axdefault  = plt.axes([pos[1,0]-width, pos[1,1]-h, width, h])  #lx,by,w,h
        #ax = plt.axes(fig, [0.3,0.11,0.9,(1-0.3)])
        self.fig = fig
        self.ax = ax
        self.X = X
        self.y = y
    
        self.w = 0. #initial point, non-array
        self.b = 0.   

        plt.show()

    
