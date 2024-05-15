import cv2
import numpy as np
from multiprocessing import Process, Value
import random



##################### Boids and Boid Behaviors ###################################################

class Boid:
    def __init__(self, x, y, size):
        self.position = np.array([x, y], dtype='float64')
        self.velocity = np.random.rand(2) * 10 - 5
        self.acceleration = np.zeros(2)
        self.size = size
        self.blackwhite = False

    def update_boids(self, swarm_window):
        if np.isnan(self.velocity).any() or np.isnan(self.position).any():
            self.position = np.array([random.randint(0, self.size-5), random.randint(0, self.size-5)], dtype='float64')
            self.velocity = np.random.rand(2) * 10 - 5  # Nulstiller position og hastighed, hvis NaN detekteres
            return

        self.velocity += self.acceleration
        self.velocity = np.clip(self.velocity, -5, 5)
        self.position += self.velocity
        self.position = np.clip(self.position, 0, self.size-5)
        self.acceleration *= 0

        if np.isnan(self.velocity).any() or np.isnan(self.position).any():
            self.position = np.array([random.randint(0, self.size-5), random.randint(0, self.size-5)], dtype='float64')
            self.velocity = np.random.rand(2) * 10 - 5  # Nulstiller position og hastighed, hvis NaN detekteres
            return

        # Draw the boid, green if drawing, blue if not
        if self.blackwhite:
            cv2.circle(swarm_window, tuple(self.position.astype(int)), 2, (0, 255, 0), -1)
        else:
            cv2.circle(swarm_window, tuple(self.position.astype(int)), 2, (255, 0, 0), -1)

    def add_trail(self, environment, trails, trails_count):
        pixel_intensity = environment[int(self.position[1]), int(self.position[0])]
        if pixel_intensity < 128:  # Tegner kun på kunst_vindue ved mørke pixels.
            trails.append(tuple(self.position.astype(int)))
            trails_count = trails_count + 1
            self.blackwhite = True
        else:
            self.blackwhite = False
        return trails_count

    def apply_behavior(self, boids, trails, beacon):                 
        separation = self.separate(boids) * 1.6                     # Makes boids seperate if they get too close
        alignment = self.align(boids) * 1.0                         # Makes boids align heading
        cohesion = self.cohere(boids) * 1.2                         # Makes boids attracted to other boids
        avoid_trails = self.avoid_trails(trails) * 0.5              # Makes boids repulsed by boid trails
        goto_beacon = self.beacon(beacon) * 0.4                     # Makes boids go to the beacon, global reach

        self.acceleration += separation + alignment + cohesion + avoid_trails + goto_beacon


    def separate(self, boids):
        desired_separation = 25
        steer = np.zeros(2)
        total = 0
        for other in boids:
            # Seperate more from non drawing boids
            if other.blackwhite:
                desired_separation = 30
            else:
                desired_separation = 40

            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < desired_separation:
                diff = self.position - other.position
                diff /= distance
                steer += diff
                total += 1
        if total > 0:
            steer /= total
            steer = steer / np.linalg.norm(steer) * 5
            steer -= self.velocity
            steer = np.clip(steer, -2, 2)
        return steer
    
    def avoid_trails(self, trails):
        desired_separation = 15
        steer = np.zeros(2)
        total = 0
        for trail in trails:
            distance = np.linalg.norm(self.position - trail)
            if 0 < distance < desired_separation:
                diff = self.position - trail
                diff /= distance
                steer += diff
                total += 1
        if total > 0:
            steer /= total
            steer = steer / np.linalg.norm(steer) * 5
            steer -= self.velocity
            steer = np.clip(steer, -2, 2)
        return steer

    def align(self, boids):

        sum_velocities = np.zeros(2)
        total = 0
        neighbor_dist = 50

        for other in boids:
            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < neighbor_dist:
                sum_velocities += other.velocity
                total += 1
        if total > 0:
            sum_velocities /= total
            sum_velocities = sum_velocities / np.linalg.norm(sum_velocities) * 5
            steer = sum_velocities - self.velocity
            steer = np.clip(steer, -2, 2)
        else:
            steer = np.zeros(2)
        return steer

    def cohere(self, boids):
        sum_positions = np.zeros(2)
        total = 0
        neighbor_dist = 60
        
        for other in boids:
            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < neighbor_dist:
                sum_positions += other.position
                total += 1
        if total > 0:
            sum_positions /= total
            direction_to_center = sum_positions - self.position
            direction_to_center = direction_to_center / np.linalg.norm(direction_to_center) * 5
            steer = direction_to_center - self.velocity
            steer = np.clip(steer, -2, 2)
        else:
            steer = np.zeros(2)
        return steer
    
    def beacon(self, beacons):
        sum_positions = np.zeros(2)
        total = 0
        neighbor_dist = 100
        
        for beacon in beacons:
            distance = np.linalg.norm(self.position - beacon)
            if 0 < distance < neighbor_dist:
                sum_positions += beacon
                total += 1
        if total > 0:
            sum_positions /= total
            direction_to_center = sum_positions - self.position
            direction_to_center = direction_to_center / np.linalg.norm(direction_to_center) * 5
            steer = direction_to_center - self.velocity
            steer = np.clip(steer, -2, 2)
        else:
            steer = np.zeros(2)
        return steer


##################### Mouse Callback Functions ###################################################

####################### Beacons #####################
beacons = []
def get_beacon(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add beacon where mouse is pressed
        beacons.append([x, y])

    # Remove beacon with mbutton
    elif event == cv2.EVENT_MBUTTONDOWN:
        if len(beacons) != 0:
            beacons.pop(0)

################### Add/Remove Boids ##################
add = [[0,0]]
remove = [[0,0]]
def add_remove_boids(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Change beacon location to where where mouse is pressed
        add.pop(0)
        add.append([x, y])

    elif event == cv2.EVENT_MBUTTONDOWN:
        remove.pop(0)
        remove.append([x, y])

################# Toggle Environment ##################
current_environment = 0
environment_number = 4
toggled  = 0
def toggle_environment(event,x,y,flags,param):
    global current_environment
    global environment_number
    global toggled
    if event == cv2.EVENT_LBUTTONDOWN:
        current_environment = current_environment + 1
        toggled = 1
    if current_environment == environment_number:
        current_environment = 0


##################### Run simulation ###################################################

def swarm_simulation(run_flag, environment_path1, environment_path2, environment_path3, environment_path4):
    # Load environments
    global current_environment
    if current_environment == 0:
        environment_path = environment_path1
    elif current_environment == 1:
        environment_path = environment_path2
    elif current_environment == 2:
        environment_path = environment_path3
    elif current_environment == 3:
        environment_path = environment_path4
    
    global toggled
    size = 300              ##################################### Set Canvas Sizes #########################
    environment = cv2.imread(environment_path, cv2.IMREAD_GRAYSCALE)
    environment = cv2.resize(environment, (size, size))

    # Start boids and other windows
    total_boids = 25        ################################# Amount of Starting Boids #####################
    boids = [Boid(random.randint(0, size-1), random.randint(0, size-1), size) for _ in range(total_boids)]
    swarm_window = np.full((size, size, 3), 255, np.uint8)
    canvas = np.zeros((size, size, 3), np.uint8)


    trails = []
    trails_count = 0

    n = 25                 ##########################3#### Amount of steps back we keep trails #############
    trails_amount = [0] * n

    while run_flag.value:
        # Reset swarm and canvas windows
        swarm_window[:] = (255, 255, 255)
        canvas[:] = (255, 255, 255) 
        if toggled:
            # Load environments
            if current_environment == 0:
                environment_path = environment_path1
            elif current_environment == 1:
                environment_path = environment_path2
            elif current_environment == 2:
                environment_path = environment_path3
            elif current_environment == 3:
                environment_path = environment_path4
            size = 300
            environment = cv2.imread(environment_path, cv2.IMREAD_GRAYSCALE)
            environment = cv2.resize(environment, (size, size))
            toggled = 0

        # Update boids ---------------------------------------------------------------
        for boid in boids:
            trails_count = boid.add_trail(environment, trails, trails_count)
            boid.apply_behavior(boids, trails, beacons)
            boid.update_boids(swarm_window)


        # Update trails --------------------------------------------------------------
        for trail in trails:
            cv2.circle(canvas, trail, 2, (0, 0, 255), -1)

        trails_amount.append(trails_count)
        #print(trails_amount)
 
        for _ in range(trails_amount.pop(0)):    
            trails.pop(0)
        
        trails_count = 0    


        # Interaction ------------------------------------------------------------
        # Draw beacon if not [0,0] (none)
        if len(beacons) != 0:
            for beacon in beacons: 
                cv2.circle(canvas, beacon, 3, (0, 255, 0), -1)

        # Add new boids where pressed
        if add[-1] != [0,0]:
            cv2.circle(swarm_window, add[-1], 3, (100, 205, 0), -1)
            boids.append(Boid(add[-1][0], add[-1][1], size))
            add.pop(0)
            add.append([0, 0])

        # Remove closest boids to press
        if remove[-1] != [0,0]:
            cv2.circle(swarm_window, remove[-1], 3, (0, 0, 255), -1)
            
            min_distance = 1000
            closest_index = None

            # Find the index of the closest coordinate
            for i, coord in enumerate(boids):
                dist = np.linalg.norm(coord.position - remove[-1])
                if dist < min_distance:
                    min_distance = dist
                    closest_index = i                

            # Pop the closest coordinate
            if closest_index is not None:
                boids.pop(closest_index)
            
            remove.pop(0)
            remove.append([0, 0])


        # Display -----------------------------------------------------------------------
        cv2.imshow('Swarm Simulation', swarm_window)
        cv2.imshow('Swarm Canvas', canvas)
        cv2.imshow('Art Input', environment)


        # Set the mouse callback function
        # Move beacon by clicking swarm simulation
        cv2.setMouseCallback('Swarm Canvas', get_beacon)
        cv2.setMouseCallback('Swarm Simulation', add_remove_boids)
        cv2.setMouseCallback('Art Input', toggle_environment)

        # Quit 
        if cv2.waitKey(50) & 0xFF == ord('q'):
            run_flag.value = False
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_flag = Value('i', 1)
    environment1_path = 'danmarkskort.jpg'
    environment2_path = 'grid.jpg'
    environment3_path = 'black_circle.jpg'
    environment4_path = '9_black_circle.png'
    swarm_process = Process(target=swarm_simulation, args=(run_flag, environment1_path, environment2_path, environment3_path, environment4_path))
    swarm_process.start()
    swarm_process.join()
