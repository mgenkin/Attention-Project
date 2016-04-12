import numpy as np
import pygame

BLACK = (0,0,0)
GREY = (100, 100, 100)
BLUE = (100, 50, 200)
GREEN = (50, 200, 100)
RED = (200, 50, 100)

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

    def accelerate(self, acc):
        # adds acceleration to velocity vector
        acc_r, acc_theta = acc
        vel_r, vel_theta = self.vel

        vel_r_new = (0.9 * vel_r) + acc_r # multiply by 0.9 for friction
        vel_theta_new = vel_theta + acc_theta
        self.vel = (vel_r_new, vel_theta_new)
        self.unitvel = (1.0, vel_theta_new)

    def move(self):
        # moves the center based on the velocity
        vel_x, vel_y = self.to_xy(self.vel)
        c_x, c_y = self.center
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
        vel_theta = self.vel[1] # we face the car in the direction of the velocity vector
        
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


if __name__ == '__main__':

    # regular pygame initializations
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Collision Detection")
    clock = pygame.time.Clock()
    done = False

    # put the car in the center of the screen
    main_car = Car((250, 250), (40, 20), (0.0, 0.0))
    # start with zero acceleration
    acc = (0.0, 0.0)

    # throw some random obstacle cars around
    obstacles = []
    for i in range(5):
        loc = np.random.random(2)*500
        obst = Car(loc.astype(int), (40, 20), (0.0, np.random.random()*(2*np.pi)))
        obstacles.append(obst)
    
    while not done:

        # --- Event Processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # exits if the window is closed
                done = True
            elif event.type == pygame.KEYDOWN:
                # Adjust speed if an arrow key is down
                if event.key == pygame.K_LEFT:
                    acc = (acc[0], -(np.pi/32.0)) # turn left
                elif event.key == pygame.K_RIGHT:
                    acc = (acc[0], (np.pi/32.0)) # turn right
                if event.key == pygame.K_UP:
                    acc = (0.5, acc[1]) # speed up
                elif event.key == pygame.K_DOWN:
                    acc = (-0.5, acc[1]) # slow down
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