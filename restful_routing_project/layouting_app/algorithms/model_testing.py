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
        self.max_height = V[2] / 2  # Tambahkan batas tinggi maksimal (setengah tinggi container)
        self.EMSs = [[np.array((0,0,0)), np.array(V)]]
        self.load_items = []  # (min_corner, max_corner, DO_index)
        
        if verbose:
            print('Init EMSs:', self.EMSs)

    def calculate_usable_volume(self):
        """Hitung volume yang bisa digunakan (dalam batas tinggi)"""
        usable_height = self.max_height
        return self.dimensions[0] * self.dimensions[1] * usable_height

    def calculate_used_volume(self):
        """Hitung volume yang sudah terisi (dalam batas tinggi)"""
        used = 0
        for item in self.load_items:
            min_c, max_c = item[0], item[1]
            if max_c[2] <= self.max_height:
                used += np.prod(max_c - min_c)
            elif min_c[2] < self.max_height:
                # Hitung partial volume jika box melewati batas
                adjusted_max = np.array([max_c[0], max_c[1], self.max_height])
                used += np.prod(adjusted_max - min_c)
        return used

    def __getitem__(self, index):
        return self.EMSs[index]

    def __len__(self):
        return len(self.EMSs)

    def update(self, box, selected_EMS, min_vol=1, min_dim=1, current_DO=None, verbose=False):
        boxToPlace = np.array(box)
        selected_min = np.array(selected_EMS[0])
        
        # Hitung tinggi yang tersedia di EMS ini
        available_height = min(selected_EMS[1][2], self.dimensions[2]) - selected_min[2]
        
        # Jika box lebih tinggi dari yang tersedia, potong box
        if boxToPlace[2] > available_height:
            if available_height < min_dim:  # Jika sisa tinggi terlalu kecil, skip
                return False
            boxToPlace[2] = available_height  # Potong box
            
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
                    # Tambahkan pengecekan tinggi maksimal untuk EMS baru
                    if new_EMS[0][2] >= self.max_height:
                        isValid = False

                    if isValid:
                        self.EMSs.append(new_EMS)
        
        return True  # Return True jika berhasil ditempatkan
    
    def calculate_usable_volume(self):
        """Hitung volume yang bisa digunakan (dalam batas tinggi)"""
        usable_height = self.max_height
        return self.dimensions[0] * self.dimensions[1] * usable_height

    def calculate_used_volume(self):
        """Hitung volume yang sudah terisi (dalam batas tinggi)"""
        used = 0
        for item in self.load_items:
            min_c, max_c = item[0], item[1]
            if max_c[2] <= self.max_height:
                used += np.prod(max_c - min_c)
            elif min_c[2] < self.max_height:
                # Hitung partial volume jika box melewati batas
                adjusted_max = np.array([max_c[0], max_c[1], self.max_height])
                used += np.prod(adjusted_max - min_c)
        return used

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
        self.original_DO_order = inputs['DOs_num']

        # Tambahkan variabel untuk partisi DO
        self.DO_partitions = {}  # {do_index: (start_x, end_x)}
        self.current_partition_x = 0
        self.partition_width = 0.2  # 20% lebar container untuk partisi awal

        # Urutkan berdasarkan: 1. Urutan DO asli 2. Random key
        indices = list(range(len(self.boxes)))
        indices.sort(key=lambda i: (
            self.original_DO_order.index(self.DOs_num[self.box_DO_map[i]]),
            solution[i]
        ))
        self.BPS = np.array(indices)
        self.VBO = solution[len(self.boxes):]

        self.num_opend_bins = 1
        self.verbose = verbose
        self.infisible = False

        self.placement()

    def placement(self):
        # Kelompokkan box berdasarkan DO
        do_groups = {}
        for i in self.BPS:
            do = self.box_DO_map[i]
            if do not in do_groups:
                do_groups[do] = []
            do_groups[do].append(i)
        
        # Urutkan DO berdasarkan urutan pengiriman asli
        sorted_DOs = sorted(do_groups.keys(), 
                          key=lambda do: self.original_DO_order.index(self.DOs_num[do]))
        
        # Hitung total volume per DO untuk menentukan alokasi partisi
        total_volume = sum(np.product(b) for b in self.boxes)
        do_volumes = {do: sum(np.product(self.boxes[i]) for i in do_groups[do]) 
                      for do in do_groups.keys()}
        
        # Buat partisi untuk setiap DO berdasarkan proporsi volume
        for do in sorted_DOs:
            partition_ratio = do_volumes[do] / total_volume
            self.DO_partitions[do] = (
                self.current_partition_x,
                self.current_partition_x + partition_ratio * self.Bins[0].dimensions[0]
            )
            self.current_partition_x += partition_ratio * self.Bins[0].dimensions[0]
        
        # Lakukan penempatan untuk setiap DO dalam partisinya
        for do in sorted_DOs:
            do_boxes = sorted(do_groups[do], key=lambda i: -np.product(self.boxes[i]))
            start_x, end_x = self.DO_partitions[do]
            
            for box_idx in do_boxes:
                box = self.boxes[box_idx]
                placed = False
                
                # Cari EMS dalam partisi DO ini
                for k in range(self.num_opend_bins):
                    EMS = self.find_EMS_in_partition(box, k, do, start_x, end_x)
                    if EMS:
                        BO = self.select_box_orientation(self.VBO[box_idx], box, EMS)
                        min_vol, min_dim = self.elimination_rule([self.boxes[i] for i in self.BPS[self.BPS > box_idx]])
                        success = self.Bins[k].update(self.orient(box, BO), EMS, min_vol, min_dim, do)
                        if success:
                            placed = True
                            break
                
                # Jika tidak ditemukan dalam partisi, cari di area umum dengan prioritas DO
                if not placed:
                    for k in range(self.num_opend_bins):
                        EMS = self.DFTRC_2_with_priority(box, k, do)
                        if EMS:
                            BO = self.select_box_orientation(self.VBO[box_idx], box, EMS)
                            min_vol, min_dim = self.elimination_rule([self.boxes[i] for i in self.BPS[self.BPS > box_idx]])
                            success = self.Bins[k].update(self.orient(box, BO), EMS, min_vol, min_dim, do)
                            if success:
                                placed = True
                                break
                
                # Jika masih belum ditempatkan, buka bin baru
                if not placed:
                    self.num_opend_bins += 1
                    if self.num_opend_bins > len(self.Bins):
                        self.infisible = True
                        return
                    EMS = self.Bins[self.num_opend_bins-1].EMSs[0]
                    BO = self.select_box_orientation(self.VBO[box_idx], box, EMS)
                    min_vol, min_dim = self.elimination_rule([self.boxes[i] for i in self.BPS[self.BPS > box_idx]])
                    self.Bins[self.num_opend_bins-1].update(self.orient(box, BO), EMS, min_vol, min_dim, do)

    def find_EMS_in_partition(self, box, bin_idx, do, start_x, end_x):
        """Cari EMS dalam partisi DO tertentu yang bisa menampung box"""
        best_EMS = None
        max_distance = -1
        
        for EMS in self.Bins[bin_idx].EMSs:
            ems_min, ems_max = EMS
            
            # Pastikan EMS berada dalam partisi DO ini
            if ems_min[0] < start_x or ems_max[0] > end_x:
                continue
                
            # Hitung jarak ke dinding belakang (DFTRC)
            distance = (self.Bins[bin_idx].dimensions[0] - ems_min[0])**2
            
            # Prioritaskan EMS dengan DO yang sama
            ems_boxes = [b for b in self.Bins[bin_idx].load_items 
                         if self.is_inside(b[0], b[1], ems_min, ems_max)]
            same_do = any(b[2] == do for b in ems_boxes)
            
            if same_do:
                distance *= 2  # Beri bonus untuk EMS dengan DO sama
                
            if distance > max_distance:
                for direction in [1,2,3,4,5,6]:
                    if self.fitin(self.orient(box, direction), EMS):
                        best_EMS = EMS
                        max_distance = distance
                        break
                        
        return best_EMS

    def DFTRC_2_with_priority(self, box, bin_idx, current_DO):
        """DFTRC dengan prioritas untuk DO yang sama"""
        best_EMS = None
        max_distance = -1
        
        for EMS in self.Bins[bin_idx].EMSs:
            ems_min, ems_max = EMS
            
            # Hitung jarak ke dinding belakang (DFTRC original)
            distance = (self.Bins[bin_idx].dimensions[0] - ems_min[0])**2
            
            # Prioritaskan EMS dengan DO yang sama
            ems_boxes = [b for b in self.Bins[bin_idx].load_items 
                         if self.is_inside(b[0], b[1], ems_min, ems_max)]
            same_do = any(b[2] == current_DO for b in ems_boxes)
            
            if same_do:
                distance *= 2  # Beri bonus untuk EMS dengan DO sama
                
            if distance > max_distance:
                for direction in [1,2,3,4,5,6]:
                    if self.fitin(self.orient(box, direction), EMS):
                        best_EMS = EMS
                        max_distance = distance
                        break
                        
        return best_EMS

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

    # def evaluate(self):
    #     if self.infisible:
    #         return INFEASIBLE
        
    #     container_volume = np.product(self.Bins[0].dimensions)
    #     is_cde = container_volume >= 350*160*160
        
    #     # Penalty sangat besar untuk multi-container di CDE
    #     if is_cde and self.num_opend_bins > 1:
    #         return 1 + (self.num_opend_bins - 1) * 10
        
    #     base_fitness = self.num_opend_bins
        
    #     # Hitung utilisasi ruang
    #     total_used = sum(np.prod(b[1]-b[0]) for bin in self.Bins[:self.num_opend_bins] for b in bin.load_items)
    #     total_available = sum(np.prod(bin.dimensions) for bin in self.Bins[:self.num_opend_bins])
    #     utilization = total_used / total_available
        
    #     # Fitness utama berdasarkan jumlah container + (1 - utilisasi)
    #     return base_fitness + (1 - utilization)
    # def get_utilization(self):
    #     total_used = sum(np.prod(b[1]-b[0]) for bin in self.Bins[:self.num_opend_bins] for b in bin.load_items)
    #     total_available = sum(np.prod(bin.dimensions) for bin in self.Bins[:self.num_opend_bins])
    #     return total_used / total_available if total_available > 0 else 0

    def get_utilization(self):
        """Hitung utilization berdasarkan volume yang bisa digunakan"""
        total_used = 0
        total_usable = 0
        for bin in self.Bins[:self.num_opend_bins]:
            total_used += bin.calculate_used_volume()
            total_usable += bin.calculate_usable_volume()
        return total_used / total_usable if total_usable > 0 else 0

    def evaluate(self):
        if self.infisible:
            return INFEASIBLE
        
        container_volume = np.product(self.Bins[0].dimensions)
        is_cde = container_volume >= 350*160*160
        
        # Hitung utilization yang benar
        utilization = self.get_utilization()
        
        # Fitness utama hanya berdasarkan jumlah container
        base_fitness = self.num_opend_bins
        
        # Berikan penalty hanya jika CDE dan multi-container
        if is_cde and self.num_opend_bins > 1:
            return base_fitness * 10  # Penalty besar untuk multi-container CDE
            
        return base_fitness + (1 - utilization)  # Beri bobot lebih kecil untuk utilizatio
    
    

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
        # Adaptasi parameter berdasarkan jumlah box
        adaptive_individuals = min(30, max(10, int(self.N/3)))  # 10-30 individu
        adaptive_generations = min(30, max(10, int(self.N/5)))  # 10-30 generasi
        
        population = np.random.uniform(0,1,(adaptive_individuals, self.num_gene))
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