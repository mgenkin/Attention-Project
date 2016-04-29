import numpy as np
import pygame

BLACK = (0,0,0)
GREY = (100, 100, 100)
BLUE = (100, 50, 200)
GREEN = (50, 200, 100)
RED = (200, 50, 100)
WHITE = (255, 255, 255)

MARGIN, COLUMNS, CARS_PER_COL = 10, 2, 6
CAR_SIZE = (100, 50)

def bw_cmap(fl, lowcol=WHITE, hicol=BLACK):
    # returns an interpolation of lowcol and hicol by fl (between zero and one)
    return [int(fl*h+(1-fl)*l) for h, l in zip(hicol, lowcol)]

class Car(object):
    """ A pygame-free Car class.
        Meant to draw using pygame's pygame.draw.polygon function, no rects involved
        Velocity and acceleration are stored in polar form, but center is in cartesian
        None of these are required to be ints, but the output of get_pointlist is converted to int for drawing
         Functions: accelerate, move, to_xy, get_pointlist
         Attributes: center, vel, size, unitvel
    """
    def __init__(self, location, size, vel):
        self.vel = vel
        self.unitvel = (1.0, vel[1]) # this is convenient for collision detection in case speed is zero
        self.center = location
        self.size = size
        self.theta_moving = 0

    def accelerate(self, acc):
        # adds acceleration to velocity vector
        acc_r, acc_theta = acc
        vel_r, vel_theta = self.vel
        vel_r_new = (0.98 * vel_r) + acc_r # multiply by 0.9 for friction
        vel_theta_new = vel_theta + acc_theta
        #if the car is moving, this stores the direction it moves in
        if abs(vel_r_new)>.01:
            self.theta_moving=vel_theta_new
        #if the car is NOT moving, it doesn't allow the car to rotate the 
        #wheels more than .7 rad, and doesn't align the car with the wheels
        #still not a really good fix for the spinning in place at all though
        if abs(vel_theta_new-self.theta_moving)>.7:
            vel_theta_new = self.theta_moving+np.sign(vel_theta_new-self.theta_moving)*.7
        self.vel = (vel_r_new, vel_theta_new)
        self.unitvel = (1.0, vel_theta_new)

    def move(self):
        # moves the center based on the velocity
        #Also doesn't let the center of the car
        #leave the screen. It just stops it. 
        #Also, not super elegant.
        screen = pygame.display.get_surface()
        [x_bound,y_bound] = screen.get_size()
        vel_x, vel_y = self.to_xy(self.vel)
        c_x, c_y = self.center
        if c_x+vel_x>x_bound or c_x+vel_x<0:
            vel_x=0
            self.vel=(0,self.vel[1])
        if c_y+vel_y>y_bound or c_y+vel_y<0:
            vel_y=0
            self.vel=(0,self.vel[1])
        self.center = (c_x + vel_x, c_y + vel_y)

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
        vel_theta = self.theta_moving # we face the car in the direction the car last moved
        
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

class Agent(Car):
    
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

    def views(self, obstacles):
        x,y  = self.to_midfront()[1] # compute distances from midpoint of front edge
        num = self.num
        views = np.zeros(num*2) # we compute views in 360 degrees, then throw away the back half
        bins = np.linspace(-np.pi, np.pi, num=2*num) # the bottoms of the angle bins
        # loop over obstacles
        for obst in obstacles:
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
                closeness = ([np.exp(-d/500.0) if d!= 0.0 else 0.0 for d in dist_current])
                # only add to the views things that are closer than what we already know is there (closer object obscure further ones)
                views = np.array([max(v, c) for v, c in zip(views, closeness)])
        # return the front half of the views, and angle centers for turning
        return views[num/2:num+num/2], [i+(4*np.pi/num) for i in bins[num/2:num+num/2]]

class ParkingLot():

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
        return obstacles

    def draw(self, surface, color=BLACK):
        surface.fill(color)
        for col in range(self.columns):
            # draw column of spaces, they are all aligned in the x direction
            x_offset = 2*self.margin+self.car_size[1]
            x = x_offset+col*(1.3*self.car_size[0]+2*self.margin+self.car_size[1]+x_offset)
            for i in range(self.cars_per_col):
                y = 1.5*self.car_size[1]+i*1.3*self.car_size[1] # turning space on top plus the height of the cars we've passed
                pygame.draw.rect(surface, RED, (x,y)+(1.3*self.car_size[0],1.3*self.car_size[1]), 2)
        return

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
    main_car = Agent((250, 250), CAR_SIZE, (0.0, 0.0), 100)
    # start with zero acceleration
    acc = (0.0, 0.0)

    # put some cars in the parking lot
    obstacles = parking_lot.make_obstacles()

    while not done:

        # --- Event Processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # exits if the window is closed
                done = True
            elif event.type == pygame.KEYDOWN:
                # Adjust speed if an arrow key is down
                if event.key == pygame.K_LEFT:
                    acc = (acc[0], -(np.pi/45.0)) # turn left
                elif event.key == pygame.K_RIGHT:
                    acc = (acc[0], (np.pi/45.0)) # turn right
                if event.key == pygame.K_UP:
                    acc = (0.1, acc[1]) # speed up
                elif event.key == pygame.K_DOWN:
                    acc = (-0.1, acc[1]) # slow down
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    acc = (0.0, acc[1]) # stop accelerating
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    acc = (acc[0], 0.0) # stop turning

        # update our car
        main_car.accelerate(acc)
        main_car.move()
        # collision detection:
        collision = False
        for obst in obstacles:
            if main_car.collision(obst):
                collision = True


        # --- Drawing
        parking_lot.draw(screen) # draw the parking lot
        main_car.draw(screen, car_color=GREEN) # draw main car
        # draw the obstacles
        for obst in obstacles:
            obst.draw(screen)
        # daw a red rectangle if there's a collision
        if collision:
            pygame.draw.rect(screen, RED, [parking_lot.size[0]-50,0,50,50])

        # Visualize the views with black/white squares on the top left
        # the closer the obstacle, the darker the square will be
        views, angles = main_car.views(obstacles)
        pygame.draw.rect(screen, BLUE, [0, 0, (2*len(views))+5, 15])
        for i, threat in enumerate(views):
            #   print threat, "threat"
            pygame.draw.rect(screen, bw_cmap(threat), [i*2, 0, 2, 10])
        
        clock.tick(60)

        # update the screen
        pygame.display.flip()

    # Close everything down
    pygame.quit()
