import math
import copy
import random
import numpy as np

INFEASIBLE = 100000

def generateInstances(N=20, m=10, V=(100,100,100)):
    def ur(lb, ub):
        value = random.uniform(lb, ub)
        return int(value) if value >= 1 else 1

    L, W, H = V
    p = []; q = []; r = []
    for i in range(N):
        p.append(ur(1/6*L, 1/4*L))
        q.append(ur(1/6*W, 1/4*W))
        r.append(ur(1/6*H, 1/4*H))

    L = [L]*m
    W = [W]*m
    H = [H]*m
    return range(N), range(m), p, q, r, L, W, H

def generateInputs(N, m, V):
    N, M, p, q, r, L, W, H = generateInstances(N, m, V)
    inputs = {'v': list(zip(p, q, r)), 'V': list(zip(L, W, H))}
    return inputs

class Bin():
    def __init__(self, V, verbose=False):
        self.dimensions = V
        self.EMSs = [[np.array((0,0,0)), np.array(V)]]
        self.load_items = []  # (min_corner, max_corner, DO_index)

        if verbose:
            print('Init EMSs:', self.EMSs)

    def __getitem__(self, index):
        return self.EMSs[index]

    def __len__(self):
        return len(self.EMSs)

    def update(self, box, selected_EMS, min_vol=1, min_dim=1, current_DO=None, verbose=False):
        boxToPlace = np.array(box)
        selected_min = np.array(selected_EMS[0])
        ems = [selected_min, selected_min + boxToPlace]
        self.load_items.append((ems[0], ems[1], current_DO))

        if verbose:
            print('------------\n*Place Box*:\nEMS:', list(map(tuple, ems)))

        for EMS in self.EMSs.copy():
            if self.overlapped(ems, EMS):
                self.eliminate(EMS)
                
                x1, y1, z1 = EMS[0]; x2, y2, z2 = EMS[1]
                x3, y3, z3 = ems[0]; x4, y4, z4 = ems[1]
                new_EMSs = [
                    [np.array((x4, y1, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y4, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y1, z4)), np.array((x2, y2, z2))]
                ]

                for new_EMS in new_EMSs:
                    new_box = new_EMS[1] - new_EMS[0]
                    isValid = True

                    for other_EMS in self.EMSs:
                        if self.inscribed(new_EMS, other_EMS):
                            isValid = False
                    if np.min(new_box) < min_dim:
                        isValid = False
                    if np.product(new_box) < min_vol:
                        isValid = False

                    if isValid:
                        self.EMSs.append(new_EMS)

    def overlapped(self, ems, EMS):
        return np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1])

    def inscribed(self, ems, EMS):
        return np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1])

    def eliminate(self, ems):
        ems = list(map(tuple, ems))
        for index, EMS in enumerate(self.EMSs):
            if ems == list(map(tuple, EMS)):
                self.EMSs.pop(index)
                return

    def get_EMSs(self):
        return list(map(lambda x: list(map(tuple, x)), self.EMSs))

    def load(self):
        return np.sum([np.product(item[1] - item[0]) for item in self.load_items]) / np.product(self.dimensions)

class PlacementProcedure():
    def __init__(self, inputs, solution, verbose=False):
        self.Bins = [Bin(V) for V in inputs['V']]
        self.boxes = inputs['v']
        self.box_DO_map = inputs['box_DO_map']
        self.DO_count = inputs['DO_count']
        self.DOs_num = inputs['DOs_num']

        indices = list(range(len(self.boxes)))
        indices.sort(key=lambda i: (self.box_DO_map[i], solution[i]))
        self.BPS = np.array(indices)
        self.VBO = solution[len(self.boxes):]

        self.num_opend_bins = 1
        self.verbose = verbose
        self.infisible = False

        self.placement()

    def placement(self):
        items_sorted = [self.boxes[i] for i in self.BPS]

        for i, box in enumerate(items_sorted):
            current_DO = self.box_DO_map[self.BPS[i]]
            selected_bin = None
            selected_EMS = None

            for k in range(self.num_opend_bins):
                EMS = self.DFTRC_2(box, k, current_DO)
                if EMS is not None:
                    selected_bin = k
                    selected_EMS = EMS
                    break

            if selected_bin is None:
                self.num_opend_bins += 1
                selected_bin = self.num_opend_bins - 1
                if self.num_opend_bins > len(self.Bins):
                    self.infisible = True
                    return
                selected_EMS = self.Bins[selected_bin].EMSs[0]

            BO = self.select_box_orientation(self.VBO[i], box, selected_EMS)
            min_vol, min_dim = self.elimination_rule(items_sorted[i+1:])
            oriented_box = self.orient(box, BO)
            self.Bins[selected_bin].update(oriented_box, selected_EMS, min_vol, min_dim, current_DO)

    def DFTRC_2(self, box, k, current_DO):
        maxDist = -1
        selectedEMS = None

        for EMS in self.Bins[k].EMSs:
            ems_min, ems_max = EMS
            for direction in [1,2,3,4,5,6]:
                d, w, h = self.orient(box, direction)
                if self.fitin((d,w,h), EMS):
                    x, y, z = ems_min
                    distance = (self.Bins[k].dimensions[0]-x-d)**2 + (self.Bins[k].dimensions[1]-y-w)**2 + (self.Bins[k].dimensions[2]-z-h)**2

                    ems_boxes = [b for b in self.Bins[k].load_items if self.is_inside(b[0], b[1], ems_min, ems_max)]
                    ems_DO_set = set([b[2] for b in ems_boxes])

                    if len(ems_DO_set) > 0 and current_DO not in ems_DO_set:
                        continue

                    if distance > maxDist:
                        maxDist = distance
                        selectedEMS = EMS

        return selectedEMS

    def is_inside(self, min1, max1, min2, max2):
        return all(min2[i] <= min1[i] and max1[i] <= max2[i] for i in range(3))

    def orient(self, box, BO=1):
        d, w, h = box
        if BO == 1: return (d, w, h)
        elif BO == 2: return (d, h, w)
        elif BO == 3: return (w, d, h)
        elif BO == 4: return (w, h, d)
        elif BO == 5: return (h, d, w)
        elif BO == 6: return (h, w, d)

    def select_box_orientation(self, VBO, box, EMS):
        BOs = []
        for direction in [1,2,3,4,5,6]:
            if self.fitin(self.orient(box, direction), EMS):
                BOs.append(direction)
        selectedBO = BOs[math.ceil(VBO*len(BOs))-1]
        return selectedBO

    def fitin(self, box, EMS):
        return all(box[d] <= EMS[1][d] - EMS[0][d] for d in range(3))

    def elimination_rule(self, remaining_boxes):
        if len(remaining_boxes) == 0:
            return 0, 0
        min_vol = 999999999
        min_dim = 9999
        for box in remaining_boxes:
            dim = np.min(box)
            if dim < min_dim:
                min_dim = dim
            vol = np.product(box)
            if vol < min_vol:
                min_vol = vol
        return min_vol, min_dim

    def evaluate(self):
        if self.infisible:
            return INFEASIBLE
        base_fitness = self.num_opend_bins
        stability_penalty = 0.0
        do_positions = {i: [] for i in range(self.DO_count)}
        for bin in self.Bins[:self.num_opend_bins]:
            for box in bin.load_items:
                min_corner, max_corner, box_DO = box
                center = (min_corner + max_corner) / 2
                do_positions[box_DO].append(center)
        for do_idx, centers in do_positions.items():
            if len(centers) == 0:
                continue
            avg_center = np.mean(np.array(centers), axis=0)
            height_center = avg_center[2]
            stability_penalty += height_center / self.Bins[0].dimensions[2]
        return base_fitness + stability_penalty

class BRKGA():
    def __init__(self, inputs, num_generations=200, num_individuals=120, num_elites=12, num_mutants=18, eliteCProb=0.7, multiProcess=False):
        self.multiProcess = multiProcess
        self.inputs = copy.deepcopy(inputs)
        self.N = len(inputs['v'])
        self.num_generations = num_generations
        self.num_individuals = int(num_individuals)
        self.num_gene = 2*self.N
        self.num_elites = int(num_elites)
        self.num_mutants = int(num_mutants)
        self.eliteCProb = eliteCProb
        self.used_bins = -1
        self.solution = None
        self.best_fitness = -1
        self.history = {'mean': [], 'min': []}

    def decoder(self, solution):
        placement = PlacementProcedure(self.inputs, solution)
        return placement.evaluate()

    def cal_fitness(self, population):
        return [self.decoder(solution) for solution in population]

    def partition(self, population, fitness_list):
        sorted_indexes = np.argsort(fitness_list)
        num_elites = min(self.num_elites, len(sorted_indexes))
        num_remaining = max(0, len(sorted_indexes) - num_elites)
        elite_indexes = sorted_indexes[:num_elites]
        non_elite_indexes = sorted_indexes[num_elites:num_elites+num_remaining]
        return (population[elite_indexes].copy(), population[non_elite_indexes].copy(), [fitness_list[i] for i in elite_indexes])

    def crossover(self, elite, non_elite):
        return [elite[gene] if np.random.uniform(0, 1) < self.eliteCProb else non_elite[gene] for gene in range(self.num_gene)]

    def mating(self, elites, non_elites):
        num_offspring = self.num_individuals - self.num_elites - self.num_mutants
        return [self.crossover(random.choice(elites), random.choice(non_elites)) for _ in range(num_offspring)]

    def mutants(self):
        return np.random.uniform(0, 1, size=(self.num_mutants, self.num_gene))

    def fit(self, patient=4, verbose=False):
        population = np.random.uniform(0,1,(self.num_individuals, self.num_gene))
        fitness_list = self.cal_fitness(population)

        best_fitness = np.min(fitness_list)
        best_solution = population[np.argmin(fitness_list)]
        self.history['min'].append(np.min(fitness_list))
        self.history['mean'].append(np.mean(fitness_list))

        best_iter = 0
        for g in range(self.num_generations):
            if g - best_iter > patient:
                self.used_bins = math.floor(best_fitness)
                self.best_fitness = best_fitness
                self.solution = best_solution
                if verbose:
                    print('Early stop at iter', g)
                return 'feasible'

            elites, non_elites, elite_fitness_list = self.partition(population, fitness_list)
            offsprings = self.mating(elites, non_elites)
            mutants = self.mutants()

            offspring = np.concatenate((mutants, offsprings), axis=0)
            offspring_fitness_list = self.cal_fitness(offspring)

            population = np.concatenate((elites, offspring), axis=0)
            fitness_list = elite_fitness_list + offspring_fitness_list

            for fitness in fitness_list:
                if fitness < best_fitness:
                    best_iter = g
                    best_fitness = fitness
                    best_solution = population[np.argmin(fitness_list)]

            self.history['min'].append(np.min(fitness_list))
            self.history['mean'].append(np.mean(fitness_list))

            if verbose:
                print("Generation:", g, "(Best Fitness:", best_fitness,")")

        self.used_bins = math.floor(best_fitness)
        self.best_fitness = best_fitness
        self.solution = best_solution
        return 'feasible'
