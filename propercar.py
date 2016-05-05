import numpy as np
import pygame

BLACK = (0,0,0)
GREY = (100, 100, 100)
BLUE = (100, 50, 200)
GREEN = (50, 200, 100)
RED = (200, 50, 100)
ACC_r = .1
ACC_t = np.pi/30
Fric = .98



class Car(object):
    """ A pygame-free Car class.
        Meant to draw using pygame's pygame.draw.polygon function, no rects involved
        Velocity and acceleration are stored in polar form, but center is in cartesian
        None of these are required to be ints, but the output of get_pointlist is converted to int for drawing
         Functions: accelerate, move, to_xy, get_pointlist
         Attributes: center, vel, size, unitvel
    """
    # size is length,width
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

        pointing = self.pointing

        vert_angle = np.arctan2(s_y, s_x); #signed angle from front to first vertex
        angles = [vert_angle, np.pi-vert_angle, vert_angle-np.pi, -vert_angle] # angles from front to each vertex
        points = []
        for ang in angles:
            disp_x, disp_y = self.to_xy((half_diag, pointing+ang)) # displacement from center to vertex
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
        vel_theta = self.pointing
        s_x = self.size[0]
        c_x, c_y = self.center
        disp_x, disp_y = self.to_xy((s_x/2.0, vel_theta))
        return [( int(c_x), int(c_y) ), ( int(c_x+disp_x), int(c_y+disp_y) )]

if __name__ == '__main__':

    # regular pygame initializations
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Collision Detection")
    clock = pygame.time.Clock()
    done = False

    # put the car in the center of the screen
    main_car = Car((250, 250), (40, 20), (0.0, 0.0))
    # start with zero acceleration and correction factor
    acc = np.zeros(2)
    correct = np.zeros(2)
    correct[1]=.3
    # throw some random obstacle cars around
    obstacles = []
    for i in range(5):
        loc = np.random.random(2)*500
        obst = Car(loc.astype(int), (40, 20), (0.0, np.random.random()*(2*np.pi)))
        obstacles.append(obst)
    
    while not done:
        # --- Event Processing
        acc = np.multiply(acc,correct)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done=True
        keys=pygame.key.get_pressed()
        if keys[pygame.K_UP]:acc[0] += ACC_r
        if keys[pygame.K_DOWN]:acc[0] += -ACC_r
        if keys[pygame.K_RIGHT]:acc[1] += ACC_t
        if keys[pygame.K_LEFT]:acc[1] += -ACC_t

        # update the position and speed of the car
        main_car.move(acc)
        # collision detection:
        collision = False
        for obst in obstacles:
            if main_car.collision(obst):
                collision = True


        # --- Drawing
        screen.fill((0,0,0)) # clear screen
        main_car.draw(screen, car_color=GREEN) # draw main car
        # draw the obstacles
        for obst in obstacles:
            obst.draw(screen)
        # daw a red rectangle if there's a collision
        if collision:
            pygame.draw.rect(screen, RED, [0,0,50,50])
        
        clock.tick(60)

        # update the screen
        pygame.display.flip()

    # Close everything down
    pygame.quit()
