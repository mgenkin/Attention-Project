import pygame
import numpy as np

GREY = (150, 150, 150)
GREEN = (34, 139, 40)
RED = (200, 50, 50)
MARGIN, COLUMNS, CARS_PER_COL = 100, 2, 6
CAR_SIZE = (100, 50)

class Obstacle(pygame.sprite.Sprite):

    def __init__(self, size, color=GREY) :
        pygame.sprite.Sprite.__init__(self)
        screen = pygame.display.get_surface()
        self.image = pygame.Surface(size)
        self.image.fill(color)
        self.rect = self.image.get_rect()

    def adjacent_vertices(self):
        # we will need adjacent vertex pairs later,
        # to calculate the binned angle inputs
        r = self.rect
        # get all vertices in counterclockwise order
        vertices = [ \
                r.topleft, \
                r.bottomleft, \
                r.bottomright, \
                r.topright \
        ]
        # put consecutive pairs of vertices into list
        adj_verts = [(vertices[i-1],vertices[i]) for i in range(4)]
        return adj_verts



class ParkingLot():

    def __init__(self, margin=100, columns=2, cars_per_col=6, car_size=CAR_SIZE):
        self.car_size = car_size
        self.columns = columns
        self.margin = margin
        self.cars_per_col = cars_per_col
        w = (2*margin+car_size[0])*columns
        h = car_size[1]*cars_per_col+2*margin
        self.size = (w,h)

    def make_map(self, obstacles):
        self.area = pygame.display.get_surface()
        self.area.fill(GREEN)
        obst_it = iter(obstacles)
        for col in range(self.columns):
            # draw column of cars, they are all aligned in the x direction
            x = self.margin+(col*(2*self.margin+self.car_size[0]))
            for i in range(self.cars_per_col):
                obst = next(obst_it)
                y = self.margin + i*self.car_size[1] # bottom margin plus the height of the cars we've passed
                if obst != None: # there is an obstacle in this parking space
                    # set car's location and draw it
                    obst.topleft = (x,y)
                    self.area.blit(obst.image, (x,y))
                # either way, draw a red rectangle to indicate a parking space
                pygame.draw.rect(self.area, RED, (x,y)+self.car_size, 2)
        return self.area

if __name__ == '__main__':

    # regular pygame initializations
    pygame.init()
    parking_lot = ParkingLot()
    screen = pygame.display.set_mode(parking_lot.size)
    pygame.display.set_caption("Basic Parking Lot")
    clock = pygame.time.Clock()
    done = False

    # randomly place cars in the parking lot
    obst = []
    for i in range(parking_lot.columns*parking_lot.cars_per_col):
        # for each parking space, randomly decide to make a car or leave it empty
        if np.random.random()>0.5:
            obst.append(Obstacle(parking_lot.car_size))
        else:
            obst.append(None)

    while not done:
        # --- Event Processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # exits if the window is closed
                done = True

        # --- Drawing


        plot_surf = parking_lot.make_map(obst)
        screen.blit(plot_surf, (0,0)) # clear screen
        clock.tick(60)

        # update the screen
        pygame.display.flip()

    # Close everything down
    pygame.quit()

