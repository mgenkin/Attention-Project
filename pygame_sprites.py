

# Import Statements
try :
    import random
    import os
    import sys
    import pygame
    import math
    import random
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
            player2.rect.inflate(-20, -20)

            if self.rect.colliderect(player1.rect) == True and not self.hit :
                angle = math.pi - angle
                self.hit = not self.hit
            elif self.rect.colliderect(player2.rect) == True and not self.hit :
                angle = math.pi - angle
                self.hit = not self.hit
            elif self.hit :
                self.hit = not self.hit
        self.vector = (angle, z)


    def calc_new_pos(self, rect, vector) :
        (angle, z) = vector # angle in radians, z is speed
        (dx, dy) = (z * math.cos(angle), z * math.sin(angle))
        return rect.move(dx,dy)

class Bat(pygame.sprite.Sprite) :
    """ Movable tennis bat to hit the ball
        Functions: reinit, update, moveup, movedown
        Attributes: which, speed"""

    STILL = 0
    MOVEUP = 1
    MOVEDOWN = 2
    LEFT = 3
    RIGHT = 4

    # Side Must be Bat.LEFT or Bat.RIGHT
    def __init__(self, side) :
        pygame.sprite.Sprite.__init__(self)
        screen = pygame.display.get_surface()
        self.image, self.rect = load_png('bat.png')
        self.area = screen.get_rect()
        self.side = side
        self.speed = 10
        self.state = Bat.STILL
        self.reinit()

    def reinit(self) :
        self.state = Bat.STILL
        self.movepos = [0, 0]
        if self.side == Bat.LEFT :
            self.rect.midleft = self.area.midleft
        elif self.side == Bat.RIGHT :
            self.rect.midright = self.area.midright

    def update(self) :
        desired_pos = self.rect.move(self.movepos)
        if self.area.contains(desired_pos) :
            self.rect = desired_pos
        pygame.event.pump()

    def moveup(self) :
        self.movepos[1] = (-1 * self.speed)
        self.state = Bat.MOVEUP

    def movedown(self) :
        self.movepos[1] = self.speed
        self.state = Bat.MOVEDOWN



def main() :
    #Initialize Screen
    pygame.init()
    screen = pygame.display.set_mode((640,480))
    pygame.display.set_caption("Basic Pong")

    #Initialize and Fill Background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0,0,0))

    #Initialize players
    global player1
    global player2
    player1 = Bat(Bat.RIGHT)
    player2 = Bat(Bat.LEFT)

    #Initialize Ball
    speed = 13
    rand = ((.1 * (random.randint(5, 8))))
    ball = Ball((0, 0), (0.47, speed))

    #Initialize Sprites
    playersprites = pygame.sprite.RenderPlain((player1, player2))
    ballsprite = pygame.sprite.RenderPlain(ball)

    #Blit background on startup
    screen.blit(background, (0,0))
    pygame.display.flip()

    #Initialize clock
    clock = pygame.time.Clock()

    #Main Event Loop
    while True :
        clock.tick(40) 

        for event in pygame.event.get() :
            if event.type == QUIT :
                sys.exit(1)
            elif event.type == KEYDOWN:
                if event.key == K_UP :
                    player1.moveup()
                elif event.key == K_DOWN :
                    player1.movedown()
                elif event.key == K_a :
                    player2.moveup()
                elif event.key == K_z :
                    player2.movedown()
            elif event.type == KEYUP :
                if event.key == K_UP or event.key == K_DOWN :
                    player1.movepos = [0,0]
                    player1.status = Bat.STILL
                elif event.key == K_z or event.key == K_a :
                    player2.movepos = [0,0] 
                    player2.staus = Bat.STILL


        screen.blit(background, ball.rect, ball.rect)
        screen.blit(background, player1.rect, player1.rect)
        screen.blit(background, player2.rect, player2.rect)
        ballsprite.update()
        playersprites.update()
        ballsprite.draw(screen)
        playersprites.draw(screen)
        pygame.display.flip()

if __name__ == "__main__" :
    main()




















