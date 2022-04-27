from .FaissKNeighbors import FaissKNeighbors
import torch
import numpy as np

class Episodic_buffer():
    '''buffer as implemented in NGU each agent holds its own buffer'''
    def __init__(self, n_neighbors=10, mu = 0.9, zeta = 0.001, epsilon = 0.0001, const = 0.001, s_m = 8):
        self.replay_buffer = []
        self.n_neighbors=n_neighbors
        #self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree')
        self.nbrs = FaissKNeighbors(n_neighbors=self.n_neighbors)
        self.mu = mu
        self.moving_average_distance = 0
        self.zeta = zeta
        self.epsilon = epsilon
        self.const = const
        self.s_m = s_m
        self.distances_list = []

    def add_state(self,state):
            self.replay_buffer.append(state.reshape(1, -1).squeeze())

    def clear(self):
        self.replay_buffer = []
        self.moving_average_distance = 0
        self.nbrs = FaissKNeighbors(n_neighbors=self.n_neighbors)
    
    def compute_new_average(self):
        for distance in self.distances_list:
            self.moving_average_distance = self.compute_EMA(self.mu,distance,self.moving_average_distance)
        self.distances_list = []
        
    def compute_episodic_intrinsic_reward(self,state):
        if len(self.replay_buffer) < 2:
            return 0
        else:
            neighbors_number = min(len(self.replay_buffer),self.n_neighbors)
            self.nbrs = FaissKNeighbors(n_neighbors=neighbors_number)
            distances, indicies = self.compute_k_nearest_neighbours_clustering(state)
            #for i in range(0, indicies.shape[1]):
            #    print(i)
            #    print('distances')
            #    print(distances[:,i])
            #    print('index')
            #    print(int(indicies[:,i].item()))
            #    print('vector difference')
            #    print(state.squeeze() - self.replay_buffer[int(indicies[:,i].item())])
                
            #compute moving average
            for i in range(0,distances.shape[1]):
                distance=distances[0,i]
                self.distances_list.append(distance)
            #prevent issue with zero moving average
            if self.moving_average_distance != 0:
                distances = distances/self.moving_average_distance
            #kill off too small distances
            if self.moving_average_distance != 0:
                distances = distances - self.zeta/self.moving_average_distance
            else: 
                distances = distances - self.zeta
            distances_too_small_indexes = np.where(distances<0)
            distances[distances_too_small_indexes] = 0
            K_v = self.epsilon / (distances+self.epsilon)
            similarity = np.sqrt(np.sum(K_v)) + self.const
            #print('similarity')
            #print(similarity)
            #if similarity >  self.s_m:
            #    print('error is in the similarity')
            #   print(similarity)
            #else:
            r_episodic = 1/similarity
            if np.isnan(r_episodic):
                print(similarity)
                print(distance)
                print(self.moving_average_distance)
                exit()
            print('similarity')
            print(similarity)
            if similarity > self.s_m:
                r_episodic = 0
            return r_episodic
    
    def compute_EMA(self,mu,x,last_average):
        """Function computes exponential decaying moving average (EMA)
        inputs: mu- parameter determining how much current data is affecting the average (i.e. the lower the more effect current data has)
                x - current data point
        last_average - past average to be updated
        outputs: new_average - updated average"""
        new_average = (1-self.mu)*x + self.mu*last_average
        return new_average

    def compute_k_nearest_neighbours_clustering(self,state):
        # check if there is more in replay buffer than number of neighbours
            # create numpy arrayx
            replay_buffer_numpy = np.stack(self.replay_buffer,axis=0)
            self.nbrs.fit(replay_buffer_numpy)
            neighbours, indices = self.nbrs.kneighbors(state.reshape(1, -1))
            #distances = np.square(neighbours)
            distances = neighbours
            return distances, indices
