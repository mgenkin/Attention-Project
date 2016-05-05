import numpy as np
import pygame
import theano
import theano.tensor as T
import lasagne
import itertools

BLACK = (0,0,0)
GREY = (100, 100, 100)
BLUE = (100, 50, 200)
GREEN = (50, 200, 100)
RED = (200, 50, 100)
WHITE = (255, 255, 255)

MARGIN, COLUMNS, CARS_PER_COL = 10, 2, 6
CAR_SIZE = (100, 50)
ANGLEBINS = 100
BATCHSIZE = 100

def bw_cmap(fl, lowcol=WHITE, hicol=BLACK):
    # returns an interpolation of lowcol and hicol by fl (between zero and one)
    return [int(fl*h+(1-fl)*l) for h, l in zip(hicol, lowcol)]

class Car(object):
    """ A pygame-free Car class.
        Meant to draw using pygame's pygame.draw.polygon function, no rects involved
        Velocity and acceleration are stored in polar form, but center is in cartesian
        None of these are required to be ints, but the output of get_pointlist is converted to int for drawing
        Also implements collision detection, which simply detects if a point of one polygon is inside the other
         Functions: accelerate, move, to_xy, inside, collision, draw, get_pointlist
         Attributes: center, vel, size, unitvel
    """
    def __init__(self, location, size, vel):
        self.vel = vel
        self.unitvel = (1.0, vel[1]) # this is convenient for collision detection in case speed is zero
        self.center = location
        self.size = size
        self.pointing = 0

    def move(self,acc):
        # Moves the car to it's new location, find the angle it's heading, based
        # on the current velocity, and the user inputted acceleration.
        c_x, c_y = self.center
        heading = self.pointing
        vel_r, vel_theta = self.vel
        acc_r, acc_theta = acc
        v_r_new = (Fric * vel_r) + acc_r # Fric is global parameter for friction
        v_t_new = .85*vel_theta + acc_theta   
        self.vel = (v_r_new,v_t_new)
        self.unitvel = (1.0, vel[1])

        # Locate the front and back "wheel" of the car
        front = np.asarray([c_x,c_y])+self.size[0]*np.asarray([np.cos(heading),np.sin(heading)])
        back = np.asarray([c_x,c_y])-self.size[0]*np.asarray([np.cos(heading),np.sin(heading)])

        #back wheel just keeps going along the direction of the car's movement, front wheel
        #goes in the direction the car's pointing+tire turned direction
        front = front+v_r_new*np.asarray([np.cos(heading+v_t_new),np.sin(heading+v_t_new)])        
        back = back+v_r_new*np.asarray([np.cos(heading),np.sin(heading)])
        #New center is just the average of the front and back in their new positions
        c_x_n, c_y_n = (front+back)/2
        #The arctan function only uses half of the available 360 deg, so this phase factor
        #extends it into the full 360. I think I could use arctan2 for this? I'm not sure
        #how to use that....
        phase=0 
        if ((front[1]-back[1])<0 and (front[0]-back[0])<0) or (front[0]-back[0]<0 and (front[1]-back[1])>0):
            phase=np.pi
        #From the position of the front and back you can get the direction the car is pointing
        self.pointing = np.arctan((front[1]-back[1])/(front[0]-back[0]))+phase

        screen = pygame.display.get_surface()
        [x_bound,y_bound] = screen.get_size()
        if c_x_n>x_bound or c_x_n<0:
            c_x_n = c_x
        if c_y_n>y_bound or c_y_n<0:
            c_y_n = c_y
        self.center = (c_x_n, c_y_n)

    def to_xy(self, r_theta):
        # converts from polar to cartesian
        r, theta = r_theta
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        return (x, y)

    def pointlist(self):
        # this determines the coordinates of the four vertices of the car, in counterclockwise order
        # the car is oriented in the direction of the velocity vector
        # plug the output of this function into pygame.draw.polygon
        s_x, s_y = self.size
        half_diag = ((s_x / 2.0)**2 + (s_y / 2.0)**2)**(0.5) # pythagorean theorem
        c_x, c_y = self.center
        vel_theta = self.pointing # we face the car in the direction the car last moved
        
        vert_angle = np.arctan2(s_y, s_x); #signed angle from front to first vertex
        angles = [vert_angle, np.pi-vert_angle, vert_angle-np.pi, -vert_angle] # angles from front to each vertex
        points = []
        for ang in angles:
            disp_x, disp_y = self.to_xy((half_diag, vel_theta+ang)) # displacement from center to vertex
            points.append( (int(c_x + disp_x), int(c_y + disp_y)) )
        return points

    def inside(self, point):
        # returns True of point is inside of the car
        p_x, p_y = point
        s_x, s_y = self.size
        c_x, c_y = self.center
        unitvel_x, unitvel_y = self.to_xy(self.unitvel) # unit velocity vector
        disp_x, disp_y = p_x-c_x, p_y-c_y # displacement of point from car center
        disp_norm = np.linalg.norm((disp_x, disp_y)) #norm of displacement vector
        disp_para = np.dot((disp_x, disp_y), (unitvel_x, unitvel_y)) # component of displacement parallel to velocity
        disp_perp = (disp_norm**2 - disp_para**2)**(0.5) # component of displacement perpendicular to velocity
        if abs(disp_para) < s_x/2.0 and abs(disp_perp) < s_y/2.0: # point is inside rectangle
            return True
        else:
            return False

    def collision(self, other):
        # returns true if one of the vertices of other is inside self, or vice versa
        other_pl = other.pointlist()
        for p in other_pl:
            if self.inside(p):
                return True
        self_pl = self.pointlist()
        for p in self_pl:
            if other.inside(p):
                return True
        return False

    def draw(self, surface, car_color=GREY, line_color=BLUE):
        # draw the car on the surface
        # draw the rectangle
        pl = self.pointlist()
        pygame.draw.polygon(surface, car_color, pl)
        # draw a line to show car's orientation
        l_start, l_end = self.to_midfront()
        pygame.draw.line(surface, line_color, l_start, l_end, 2)

    def to_midfront(self):
        # gives a list containing the center and the midpoint of the front side
        # I use this to draw the blue line so you can see which way the car is turned
        vel_theta = self.vel[1]
        s_x = self.size[0]
        c_x, c_y = self.center
        disp_x, disp_y = self.to_xy((s_x/2.0, vel_theta))
        return [( int(c_x), int(c_y) ), ( int(c_x+disp_x), int(c_y+disp_y) )]

class ParkingSpace(Car):
    """ A parking space, same as a car
    """
    def __init__(self, location, size, vel):
        super(ParkingSpace,self).__init__(location, size, vel)

    def draw(self, surface, car_color=BLUE):
        # draw the car on the surface
        # draw the rectangle
        pl = self.pointlist()
        pygame.draw.polygon(surface, car_color, pl) 

class Agent(Car):
    """ The agent, contains functionality for moving and computing "vision" vector.
        Subclasses the Car class.
         Functions: signed_angle, views
    """
    
    def __init__(self, location, size, vel, num_anglebins):
        Car.__init__(self, location, size, vel)
        self.num = num_anglebins

    def signed_angle(self, v1, v2):
        # signed angle between two vectors
        # the angle between [1,0] and [0,1] is 0.5*pi
        # the angle between [0,1] and [1,0] is -0.5*pi
        ang = (np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))
        if ang > np.pi:
            return 2*np.pi-ang
        elif ang < -np.pi:
            return 2*np.pi+ang
        else:
            return ang

    def views(self, obstacles, spaces):
        x,y  = self.to_midfront()[1] # compute distances from midpoint of front edge
        num = self.num
        views = np.zeros(num*2) # we compute views in 360 degrees, then throw away the back half
        bins = np.linspace(-np.pi, np.pi, num=2*num) # the bottoms of the angle bins
        # loop over obstacles
        for obst in itertools.chain(obstacles, spaces):
            # loop over consecutive vertex pairs for each obstacle
            pl = obst.pointlist()
            for pl1,pl2 in [(pl[i-1], pl[i]) for i in range(len(pl))]:
                
                diff1 = pl1[0] - x, pl1[1] - y # difference vector
                dist1 = np.linalg.norm(diff1) # euclidean distance from agent to obstacle
                theta1 = self.signed_angle(self.to_xy(self.unitvel), diff1) # angle from velocity vector
                bin1 = len(filter(lambda x: x<theta1, bins)) # the bin that the angle belongs in

                diff2 = pl2[0] - x, pl2[1] - y
                theta2 = self.signed_angle(self.to_xy(self.unitvel), diff2)
                dist2 = np.linalg.norm(diff2)
                bin2 = len(filter(lambda x: x<theta2, bins))

                dist_current = np.zeros(views.shape)
                
                if abs(bin2-bin1) > 1:
                    # linearly interpolate the distance value between the two bins
                    if bin2>bin1:
                        dist_current[bin1:bin2+1] = np.linspace(dist1, dist2, num = (bin2+1-bin1))
                    else:
                        dist_current[bin2:bin1+1] = np.linspace(dist2, dist1, num = (bin1+1-bin2))
                elif abs(bin2-bin1) == 1:
                    # simply fill the distances into the bins
                    dist_current[bin1] = dist1
                    dist_current[bin2] = dist2
                else: # bin2 == bin1
                    # count the closer one.
                    dist_current[bin2] = min(dist2, dist1)
                # calculate our closeness function
                if type(obst) is ParkingSpace: # it's a parking space
                    closeness = ([np.exp(-d/500.0) if d!= 0.0 else 0.0 for d in dist_current])
                else : # it's an empty space
                    closeness = ([-np.exp(-d/500.0) if d!= 0.0 else 0.0 for d in dist_current])
                # only add to the views things that are closer than what we already know is there (closer object obscure further ones)
                views = np.array([max(v, c, key=lambda t:abs(t)) for v, c in zip(views, closeness)])
        # return the front half of the views, and angle centers for turning
        return views[num/2:num+num/2], [i+(4*np.pi/num) for i in bins[num/2:num+num/2]]

class ParkingLot():
    """ Builds a parking lot and draws it, with parking spaces.
        Has a function to generate stationary car-sized obstacles in some of the parking spaces
         Functions: make_obstacles, draw
         Attributes: car_size, columns, margin, cars_per_col, size

    """
    def __init__(self, margin=100, columns=2, cars_per_col=6, car_size=CAR_SIZE):
        self.car_size = car_size
        self.columns = columns
        self.margin = margin
        self.cars_per_col = cars_per_col
        #margin is the space between the car and the objects we want it to be moving past
        #Each column has a car's width of space on either side, with margins
        #the tops and bottoms are 1.5 car widths wide, to allow for turning the car around
        #the tops of the columns. The parking spaces are 1.3:1 of the cars dimensions. 
        #which is all approximately based on real car dimensions. 
        w = (4*margin+1.3*car_size[0]+2*car_size[1])*columns
        h = (1.3*cars_per_col+3)*car_size[1]
        self.size = (int(w),int(h))

    def make_obstacles(self):
        obstacles = []
        spaces = []
        diff_x, diff_y = self.car_size[0]/2.0, self.car_size[1]/2.0 # to adjust from top left corner to center of car
        for col in range(self.columns):
            # loop over x,y of parking spaces
            x_offset = 2*self.margin+self.car_size[1]+.15*self.car_size[0]
            x = x_offset+col*(1.15*self.car_size[0]+2+2*self.margin+self.car_size[1]+x_offset)
            for i in range(self.cars_per_col):
                y = 1.65*self.car_size[1]+i*1.3*self.car_size[1] # bottom margin plus the height of the cars we've passed
                if np.random.random()>0.5: # randomly decide if there will be an obstacle
                    obst = Car((x+diff_x, y+diff_y), CAR_SIZE, (0.0, 0.0))
                    obstacles.append(obst)
                else:
                    # make a parking space
                    space = ParkingSpace((x+diff_x, y+diff_y), CAR_SIZE, (0.0, 0.0))
                    spaces.append(space)
        return obstacles, spaces

    def draw(self, surface, color=BLACK):
        # draw the map on the surface
        surface.fill(color)
        for col in range(self.columns):
            # draw column of spaces, they are all aligned in the x direction
            x_offset = 2*self.margin+self.car_size[1]
            x = x_offset+col*(1.3*self.car_size[0]+2*self.margin+self.car_size[1]+x_offset)
            for i in range(self.cars_per_col):
                y = 1.5*self.car_size[1]+i*1.3*self.car_size[1] # turning space on top plus the height of the cars we've passed
                pygame.draw.rect(surface, RED, (x,y)+(1.3*self.car_size[0],1.3*self.car_size[1]), 2)
        return

# LEARNING MODEL
class NN(object):
    """ General neural network class, based on theano and lasagne
         Attributes: invar, outvar, inshape, outshape, batchsize, X, y, counter
         Functions: build_network, build_training_functions, add_data, train, best_action
    """
    def __init__(self, inshape, outshape, batchsize):
        self.inshape, self.outshape, self.batchsize = inshape, outshape, batchsize
        self.invar = T.matrix()
        self.outvar = T.vector()
        self.lrvar = T.scalar()
        self.build_network(inshape[0], outshape[0], input_var=self.invar)
        self.build_training_functions(self.invar, self.outvar, self.lrvar)
        self.X = np.zeros((batchsize,)+inshape)
        self.y = np.zeros((batchsize,)+outshape)
        self.counter = 0

    def build_network(self, inshape, outshape, input_var = None):
        # create neural network in lasagne, check out lasagne's mnist example if this is weird
        # or just skip it for now =)
        print inshape
        l_in = lasagne.layers.InputLayer(shape=(None, inshape),
                                         input_var=input_var)
        l_hid1 = lasagne.layers.DenseLayer(
                l_in, num_units=inshape,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        l_out = lasagne.layers.DenseLayer(
                l_hid1, num_units=outshape,
                nonlinearity=lasagne.nonlinearities.sigmoid)
        self.nn = l_out

    def build_training_functions(self, input_var, target_var, lr_var):
        # compile training functions with learning rate
        network = self.nn
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=lr_var, momentum=0.9)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)

        self.train_fn = theano.function([input_var, target_var, lr_var], loss, updates=updates)
        self.pred_fn = theano.function([input_var], test_prediction)

    def add_data(self, x, y):
        # add the training data for the neural network
        # the counter goes from 0 to self.batchsize over and over again, updating the same minibatch
        counter = self.counter
        self.X[counter, ...] = x
        self.y[counter, ...] = y
        counter += 1
        counter %= self.batchsize
        self.counter=counter

    def train(self):
        # go once through the current batch
        cost = self.train_fn(X, y)
        return cost

    def best_action(self, state, actionshape):
        action = np.zeros(actionshape)
        pred_cost = []
        for i in range(actionshape):
            # predict the cost of doing each action
            action[i] = 1
            pred_cost.append((action.copy(), self.pred_fn(np.concatenate((state, action))[np.newaxis,:])))
            action[i] = 0
        # output the action with minimum predicted cost
        return min(pred_cost, key=lambda t: t[1])

def cost(collision):
    if collision: 
        return -100.0
    else:
        # find intersection area with parking space
        return 0.0

if __name__ == '__main__':

    # make the parking lot
    parking_lot = ParkingLot(margin=MARGIN, columns=COLUMNS, cars_per_col=CARS_PER_COL, car_size=CAR_SIZE)

    # regular pygame initializations
    pygame.init()
    screen = pygame.display.set_mode(parking_lot.size)
    pygame.display.set_caption("Parking Lot")
    clock = pygame.time.Clock()
    done = False

    # put the car in the center of the screen
    main_car = Agent((250, 250), CAR_SIZE, (0.0, 0.0), ANGLEBINS)
    # start with zero acceleration
    acc = (0.0, 0.0)

    # put some cars and empty spaces
    obstacles, spaces = parking_lot.make_obstacles()

    # initialize the neural network
    nn = NN((3*ANGLEBINS,), (1,), BATCHSIZE)

    while not done:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # exits if the window is closed
                done = True

        # get car's vision input and angles for turning
        views, angles = main_car.views(obstacles, spaces)
        # get the action with minimum predicted cost from neural network
        action, pred_cost = nn.best_action(views, 2*ANGLEBINS)
        # the action is an array of size 2*ANGLEBINS
        # the first ANGLEBINS indicates moving forward in that direction
        # the second ANGLEBINS indicates moving backward in that direction
        act_one = np.argmax(action)
        theta_ind, dir_ind = act_one%BATCHSIZE, act_one/BATCHSIZE
        acc = (dir_ind if dir_ind>0 else -1), angles[theta_ind]

        # update our car
        main_car.move(acc)
        # collision detection:
        collision = False
        for obst in obstacles:
            # collision with each obstacle
            if main_car.collision(obst):
                collision = True

        # --- Drawing
        parking_lot.draw(screen) # draw the parking lot
        main_car.draw(screen, car_color=GREEN) # draw main car
        # draw the obstacles
        for obst in obstacles:
            obst.draw(screen)
        # draw the parking spaces
        for space in spaces:
            space.draw(screen)
        # daw a red rectangle if there's a collision
        if collision:
            pygame.draw.rect(screen, RED, [parking_lot.size[0]-10,0,10,10])

        # Visualize the views with black/white squares on the top left
        # the closer the obstacle, the darker the square will be
        pygame.draw.rect(screen, BLUE, [0, 0, (2*len(views))+5, 15])
        for i, threat in enumerate(views):
            #   print threat, "threat"
            pygame.draw.rect(screen, bw_cmap((threat+1.0)/2, hicol=GREEN, lowcol=RED), [i*2, 0, 2, 10])
        
        clock.tick(10)

        # update the screen
        pygame.display.flip()

        # record the data 
        true_cost = cost(collision)
        nn.add_data(np.concatenate((views,action)), true_cost)

    # Close everything down
    pygame.quit()
