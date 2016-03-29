import pygame
import numpy as np

# define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# norm of velocity vector
VEL_NORM = 3.0
# how many views
ANGLE_BINS = 15
# number of obstacles
NUM_OBSTACLES = 30
# screen size
SIZE = [500, 500]
# costs for distances from obstacle
# for (dist, cost) if DIST_COST, 
#  if the agent is within dist pixels of an obstacle, it incurs the corresponding cost
# this is visualized by the rings around the obstacles: the darker the ring, the higher the cost 
DIST_COST = [(10, 1.0), (20, 0.75), (30, 0.5), (40, 0.25), (50, 0.1)]
# allowed radius for erasing obstacles with the mouse
OVERLAP_DIST = 30

def signed_angle(v1, v2):
    # signed angle between two vectors
    # the angle between [1,0] and [0,1] is 0.5*pi
    # the angle between [0,1] and [1,0] is -0.5*pi
    return np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])

def binned_angle_views(agent_pos, agent_vel, obstacle_locations, num=ANGLE_BINS, size=SIZE):
    views = [0.0]*num
    # this corresponds to angle_bins divided by pi
    bins_n = np.linspace(-0.5, 0.5, num=num)
    angle_bins = [i*np.pi for i in bins_n]
    # normalize our velocity vector, to compute angles later on
    vel_n = np.array(agent_vel)/np.linalg.norm(agent_vel)
    # loop over the obstacles to fill in the views
    for obst in obstacle_locations:
        # compute the difference vector
        distance, diff = toroidal_difference(agent_pos, obst, size)
        # normalize vector to compute the angles
        diff_n = diff/distance
        # angle between vectors divided by pi
        angle_n = signed_angle(vel_n, diff_n)/np.pi
        if abs(angle_n) >0.5:
            # if agent is facing away from obstacle, ignore
            continue
        else:
            # to find which view the obstacle belongs in,
            # find how many of our lower bounds it is above
            # (ask me if this is not clear)
            bin = len(filter(lambda x: x<angle_n, bins_n))
            # compute closeness level, ignore things further than 50 away
            closeness = 1.0-min(distance/50.0, 1.0)
            if views[bin] > closeness:
                # if there's already a closer one, whatever
                continue
            else:
                # otherwise this is the one we care about
                views[bin] = closeness
    # return views and centers of angle bins
    return views, [i+(4*np.pi/num) for i in angle_bins] 
    
def toroidal_difference(agent, obstacle, size):
    # gives proper distance, considering how the 
    # edges are handled.  This code is easier explained
    # with a picture, so ask me
    x1, y1 = obstacle
    x2, y2 = agent
    xs, ys = size
    possible_xy = [(x1, y1), (x1+xs, y1), (x1-xs, y1), \
                (x1, y1+ys), (x1+xs, y1+ys), (x1-xs, y1+ys),\
                (x1, y1-ys), (x1+xs, y1-ys), (x1-xs, y1-ys)]
    diffs = [(xi-x2, yi-y2) for (xi, yi) in possible_xy]
    with_distances = [(np.linalg.norm([x,y]), (x, y)) for (x,y) in diffs]
    return sorted(with_distances, key=lambda x:x[0])[0]

def vector_from_angle(theta, from_vec, norm=1.0, truncate=True):
    # gives vector facing in the direction of angle theta w/r/t from_vec
    theta_from_vec = theta + signed_angle([1,0], from_vec)
    vec = [np.cos(theta_from_vec)*norm, np.sin(theta_from_vec)*norm]
    if truncate:
        # truncate so we can use it in pygame
        vec = [int(i) for i in vec]
    return vec

def bw_cmap(fl, lowcol=WHITE, hicol=BLACK):
    # returns an interpolation of lowcol and hicol by fl (between zero and one)
    return [int(fl*h+(1-fl)*l) for h, l in zip(hicol, lowcol)]

def next_vel(views, angle_bin_centers, agent_vel):
    # choose a direction that's at right angles to the closest obstacle
    worst_direction = max(list(enumerate(views)), key=lambda x: x[1])[0]
    # approximately orthogonal direction
    num = len(views)
    direction = (worst_direction + (num/2)) % num
    return vector_from_angle(angle_bin_centers[direction], agent_vel, norm=VEL_NORM), direction

if __name__ == '__main__':

    # regular pygame initializations
    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("First Attention Model")
    done = False
    clock = pygame.time.Clock()
     
    # Starting position and velocity of the agent
    agent_pos = [250, 250]
    agent_vel = vector_from_angle(0.0, [0.0,1.0], norm=VEL_NORM)
    # Set random obstacle locations
    obstacle_locations = []
    for i in range(NUM_OBSTACLES):
        obstacle_locations.append([int(np.random.random()*SIZE[0]), int(np.random.random()*SIZE[0])])
    
     # -------- Main Program Loop -----------
    while not done:
        # --- Event Processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # exits if the window is closed
                done = True
            if event.type == pygame.MOUSEBUTTONUP:
                # mouse is clicked
                pos = list(pygame.mouse.get_pos())
                # if there's an obstacle near there, remove it
                removal_index = None
                for ind, obst in enumerate(obstacle_locations):
                    if np.linalg.norm(np.array(obst)-np.array(pos))<OVERLAP_DIST:
                        removal_index = ind
                if removal_index != None:
                    obstacle_locations.pop(removal_index)
                else:
                    # otherwise, add an obstacle there
                    obstacle_locations.append(pos)

        # --- Logic
        # Move the rectangle starting point
        agent_pos = [(pos+vel)%s for pos, vel, s in zip(agent_pos, agent_vel, SIZE)]
        # agent_facing isn't used, it's just for drawing convenience later on
        agent_facing = [pos+(2*vel) for pos, vel in zip(agent_pos, agent_vel)]
        # compute the closeness of closest obstacle in each angle bin
        views, angle_bin_centers = binned_angle_views(agent_pos, agent_vel, obstacle_locations)
        # decide where to move next
        agent_vel, chosen_direction = next_vel(views, angle_bin_centers, agent_vel)
        
        # --- Drawing
        # Set the screen background
        screen.fill(WHITE)
        # Draw the obstacles
        for obst in obstacle_locations:
            for dist, cost in sorted(DIST_COST, key=lambda t:t[1]):
                pygame.draw.circle(screen, bw_cmap(cost, lowcol=WHITE, hicol=RED), obst, dist)

        # Draw the agent with a line to show direction
        pygame.draw.rect(screen, BLACK, [i-2 for i in agent_pos]+[4,4])
        pygame.draw.line(screen, BLUE, agent_pos, agent_facing, 2)

        # Visualize the views with black/white squares on the top left
        # the closer the obstacle, the darker the square will be
        pygame.draw.rect(screen, BLUE, [0, 0, (10*len(views))+5, 15])
        for i, threat in enumerate(views):
            pygame.draw.rect(screen, bw_cmap(threat), [i*10, 0, 10, 10])

        # --- Wrap-up
        # Limit to 60 frames per second
        clock.tick(60)
     
        # update the screen
        pygame.display.flip()
     
    # Close everything down
    pygame.quit()
