

# Import Statements
try :
    import random
    import os
    import sys
    import pygame
    import math
    import random
    import numpy as np
    import getopt
    from socket import *
    from pygame.locals import *
except ImportError as err :
    print "There was an error while importing."
    print err
    sys.exit(1)
#------------------------End Import Statements----------------------------------

# Resource Handling Functions
def load_png(image_name) :
    """Load image and return image object"""
    full_path = os.path.join(os.getcwd() , 'images', image_name)

    try :
        image = pygame.image.load(full_path)
        if image.get_alpha() == None :
            image = image.convert()
        else :
            image = image.convert_alpha()
    except pygame.error as err :
        raise SystemExit('Failed to load the image %s' % full_path)

    return image, image.get_rect()

#-------------------End Resource Handling Functions-----------------------------

# Class Definitions

BLUE = (50, 50, 200)

class Ball(pygame.sprite.Sprite) :
    """ A ball that will move across the screen.
         Functions: update, calc_new_pos
         Attributes: area, vector
    """

    def __init__(self, (xy), vector) :
        pygame.sprite.Sprite.__init__(self)
        self.image, self.rect = load_png('ball.png')
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.vector = vector
        self.hit = False

    def update(self) :
        new_pos = self.calc_new_pos(self.rect, self.vector)
        self.rect = new_pos
        (angle, z) = self.vector
        if not self.area.contains(new_pos) :
            tl = not self.area.collidepoint(new_pos.topleft)
            tr = not self.area.collidepoint(new_pos.topright)
            bl = not self.area.collidepoint(new_pos.bottomleft)
            br = not self.area.collidepoint(new_pos.bottomright)

            if tl and tr or (bl and br) :
                angle = -angle
            if tl and bl :
                #self.offcourt(player = 2)
                angle = math.pi - angle
            if tr and br :
                #self.offcourt(player = 3)
                angle = math.pi - angle
        else :
            # Deflate the rectangles so you cant catch ball behind bat
            player1.rect.inflate(-20, -20)

            if self.rect.colliderect(player1.rect) == True and not self.hit :
                angle = math.pi - angle
                self.hit = not self.hit
            elif self.hit :
                self.hit = not self.hit
        self.vector = (angle, z)


    def calc_new_pos(self, rect, vector) :
        (angle, z) = vector # angle in radians, z is speed
        (dx, dy) = (z * math.cos(angle), z * math.sin(angle))
        return rect.move(dx,dy)

class Car(pygame.sprite.Sprite) :
    """ Movable car to drive around
        Functions: reinit, update, moveup, movedown
        Attributes: which, speed"""

    maintain = 0
    forward = 1
    backward = -1
    LEFT = -5
    RIGHT = 5

    # Side Must be Bat.LEFT or Bat.RIGHT
    def __init__(self, size, loc, color=BLUE) :
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface(size)
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.vel = (0.0, 0.0)
        self.rect.center = loc

    def update(self) :
        screen = pygame.display.get_surface()
        [x_bound,y_bound] = screen.get_size()
        [dx,dy]=self.to_xy(self.vel[0], self.vel[1])
        if self.rect.right+dx>x_bound or self.rect.left+dx<0:
            dx=0
        if self.rect.bottom+dy>y_bound or self.rect.top+dy<0:
            dy=0
        self.rect.move_ip([dx,dy])
        pygame.event.pump()

    def to_xy(self, r, theta):
        x = int(r*np.cos(math.pi*theta/180))
        y = int(r*np.sin(math.pi*theta/180))
        return (x, y)

    def update_vel(self, acc):
        r, theta = self.vel
        r = 0.9*r + acc[0]
        theta = theta + acc[1]
        self.vel = (r, theta)



def main() :
    #Initialize Screen
    pygame.init()
    screen_w=640
    screen_h=480
    screen = pygame.display.set_mode([screen_w,screen_h])
    pygame.display.set_caption("Drive Me!")

    #Initialize and Fill Background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0,0,0))

    #Initialize players
    global player1
    player1 = Car((20, 10), (250, 250))

    #Initialize Ball
    speed = 13
    rand = ((.1 * (random.randint(5, 8))))
    ball = Ball((0, 0), (0.47, speed))

    #Initialize Sprites
    playersprites = pygame.sprite.RenderPlain(player1)
    ballsprite = pygame.sprite.RenderPlain(ball)

    #Blit background on startup
    screen.blit(background, (0,0))
    pygame.display.flip()

    #Initialize clock
    clock = pygame.time.Clock()
    update_last = np.zeros(2)
    state=0
    #Main EventLoop
    while True :
        clock.tick(40) 
        update = np.zeros(2)
        for event in pygame.event.get():
        	if event.type==pygame.QUIT:sys.exit()
        keys=pygame.key.get_pressed()
        if keys[pygame.K_UP]:update[0] += 1
        if keys[pygame.K_DOWN]:update[0] += -1
        if keys[pygame.K_RIGHT]:update[1] += 5
        if keys[pygame.K_LEFT]:update[1] += -5        
        player1.update_vel(update)
        screen.blit(background, ball.rect, ball.rect)
        screen.blit(background, player1.rect, player1.rect)
        player1.update()
        ballsprite.update()
        playersprites.update()
        ballsprite.draw(screen)
        playersprites.draw(screen)
        pygame.display.flip()

if __name__ == "__main__" :
    main()
