# Make sprites without using image files
# @author Brian McEntee

import pygame

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

BACKGROUND_COLOR = WHITE

class Block(pygame.sprite.Sprite) :

    def __init__(self, color, width, height) :
        #Call parent class Sprite constructor
        pygame.sprite.Sprite.__init__(self)

        #Get area of the current game screen
        curr_screen = pygame.display.get_surface()
        self.area = curr_screen.get_rect()
        self.speed = [2,2]
        self.radius = width / 2
        #Create an image of the block and fill it with color
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        #Fetch the rectangle object that has the dimensions of the image
        #Update position by setting rect.x and rect.y to desired value
        self.rect = self.image.get_rect()

        #Draw image of ball
        pygame.draw.circle(self.image, BLACK, self.rect.center, self.radius)

    def update(self) :
        # update current position of rectangle
        self.rect = self.rect.move(self.speed)
        # if position is off screen reverse direction
        if self.rect.left < 0 or self.rect.right > self.area.right :
            self.speed[0] *= -1
        if self.rect.top < 0 or self.rect.bottom > self.area.bottom :
            self.speed[1] *= -1 


def main () :

    pygame.init()

    # Set width and height of the screen
    size = (700, 500)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("The Block")

    # Initialize and fill background
    background = pygame.Surface(screen.get_size())
    background.convert()
    background.fill(WHITE)

    # Initialize Block
    block = Block(WHITE, 50, 50)

    # Initialize sprite
    blocksprite = pygame.sprite.RenderPlain(block)

    # Blit background onto screen on startup
    screen.blit(background, [0,0])
    pygame.display.flip()

    # Loop until user clicks the closed button
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()


    # --------- Main Program Loop -------------
    while not done :

        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                done = True

        # ----- Game Logic Goes Here -----
        # ----- Screen Clearing Code Goes Here -----
        screen.blit(background, block.rect, block.rect)
        block.update()
        # ----- Drawing Code Goes Here -----
        blocksprite.draw(screen)
        # ----- Update Screen With What Was Drawn
        pygame.display.flip()

        # ----- Limit to 60 frames a second
        clock.tick(60)

    # Close the window and quit
    pygame.quit()

if __name__ == "__main__" :
    main()