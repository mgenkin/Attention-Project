import pygame
import numpy as np

GREY = (150, 150, 150)
GREEN = (34, 139, 40)
RED = (200, 50, 50)
MARGIN, COLUMNS, CARS_PER_COL = 100, 2, 6
CAR_SIZE = (100, 50)

class Rect_obj(object):
	# any rectangular object should subclass this, including 
	# the obstacles and the agent
	def __init__(self, top_left, size=None):
		
		self.top_left = top_left
		if size != None:
			self.size = size
		else:
			self.size = CAR_SIZE

	def adjacent_vertices(self):
		# we will need adjacent vertex pairs later,
		# to calculate the binned angle inputs
		x, y = self.top_left
		w, h = self.size
		# get all vertices in counterclockwise order
		vertices = []
		for i,j in [(0, 0), (0, 1), (1,1), (1,0)]:
			vertices.append((x+i*w, y+j*h))
		# put consecutive pairs of vertices into list
		adj_verts = [(vertices[i-1], vertices[i]) for i in range(4)]
		return adj_verts

	def draw(self, surface, color=GREY, border_color=RED):
		tl_x, tl_y = self.top_left
		s_x, s_y = self.size
		# draw the border
		pygame.draw.rect(surface, border_color, [tl_x, tl_y, s_x, s_y], 2)
		# fill the inside
		surface.fill(color, rect=[tl_x+2, tl_y+2, s_x-3, s_y-3])
		

def map_size(margin=100, columns=2, cars_per_col=6, car_size=CAR_SIZE):
	# gives the size that the map should be, if using the make_map function below
	car_w, car_h = car_size
	w = (2*margin+car_w)*columns
	h = car_h*cars_per_col+2*margin
	return (w,h)

def make_map(surface, cars, margin=100, columns=2, cars_per_col=6, car_size=CAR_SIZE):
	# the margin is the width on the left and right of every parking space
	# it is also the height of the space on the bottom and top of the column
	surface.fill(GREEN)
	car_w, car_h = car_size
	cars_it = iter(cars)
	for col in range(columns):
		# draw column of cars, they are all aligned in the x direction
		x = margin+(col*(2*margin+car_w))
		for i in range(cars_per_col):
			car = next(cars_it)
			y = margin + i*car_h # bottom margin plus the height of the cars we've passed
			if car != None:
				# set car's location and draw it
				car.top_left = x, y
				car.draw(surface)
			else:
				# draw an empty rectangle if there's no car
				pygame.draw.rect(surface, RED, [x,y,car_w,car_h], 2)
	return


if __name__ == '__main__':
	# regular pygame initializations
	pygame.init()
	screen = pygame.display.set_mode(map_size(margin=MARGIN, columns=COLUMNS, cars_per_col=CARS_PER_COL))
	print map_size(), " map size"
	pygame.display.set_caption("Basic Map")
	clock = pygame.time.Clock()
	done = False

	# randomly place cars in the parking lot
	cars = []
	for i in range(COLUMNS*CARS_PER_COL):
		# for each parking space, randomly decide to make a car or leave it empty
		if np.random.random()>0.5:
			cars.append(Rect_obj(None, size=CAR_SIZE))
		else:
			cars.append(None)

	while not done:
		# --- Event Processing
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				# exits if the window is closed
				done = True

		# --- Drawing
		# draw the map
		make_map(screen, cars)
		clock.tick(60)

		# update the screen
		pygame.display.flip()

	# Close everything down
	pygame.quit()

