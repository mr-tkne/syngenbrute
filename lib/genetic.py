import traceback
import sys
import math
from random import random, randint
import hashlib

class Speciman:
    def __init__(self, params, id):
        self.id = id
        self.values = [int(param['min'] + (param['max'] - param['min']) * random())
                       for param in params]
        self.score = None

    def hash(self):
        hash = hashlib.md5()
        hash.update("".join(str(int(value)) for value in self.values).encode())
        # print(hash, self.values)
        return hash.hexdigest()


class GeneticModel:
    def __init__(self, pop_size=20, mutation_ratio=.25, mutation_decline=0.98,
                 epoch=100, adjust_limits_each_epoch=True):
        self.test_results = {}
        self.maxleaders = 5
        self.params = []
        self.generations = []
        self.specimen = []
        self.leaders = []
        self.hiscore = 1e-10
        self._sid = 0
        self._gid = 0
        self.pop_size = pop_size
        self.mutation_ratio = mutation_ratio
        self.original_mutation_ratio = mutation_ratio
        self.epoch = epoch
        self.adjust_limits_each_epoch = adjust_limits_each_epoch
        self.mutation_decline = mutation_decline
        self.test_function = lambda speciman: 1.0

    def add_param(self, key, min, max, mutation_mult=1.0):
        self.params.append({
            "key": key,
            "origmin": min,
            "origmax": max,
            "min": min,
            "max": max,
            "mutation_mult": mutation_mult
            })

    def init_generation(self):
        gen = [Speciman(self.params, self._sid + i)
               for i in range(self.pop_size)]
        self.specimen += gen
        self.generations += [gen]
        genid = self._gid
        self._gid += 1
        self._sid += self.pop_size
        return (gen, genid)

    def getLeaders(self):
        res = []
        for (score, id) in self.leaders:
            res.append((score, self.specimen[id]))
        return res

    def rate(self, speciman_id, score):
        self.specimen[speciman_id].score = score
        if len(self.leaders) < 2 or score > self.leaders[-1][0]:
            self.leaders.append((score, speciman_id))
            self.leaders.sort(reverse=True)
            self.leaders = self.leaders[0:self.maxleaders]
            self.hiscore = self.leaders[0][0]

    def crossover(self):
        gen = []
        # top = [self.specimen[self.leaders[0][1]],
        #        self.specimen[self.leaders[1][1]]
        #        ]
        top = [self.specimen[self.leaders[i][1]]
               for i in range(len(self.leaders))]
        random_size = int(self.pop_size/(1.61**(self._gid % self.epoch)))
        clones_size = int(0.2 * (self.pop_size - random_size))
        crossover_size = self.pop_size - random_size - clones_size
        for i in range(crossover_size):
            speciman = Speciman(self.params, self._sid + i)
            for j in range(len(self.params)):
                gene_source = randint(0, len(top)-1)
                speciman.values[j] = int(top[gene_source].values[j])
            gen.append(speciman)
        self._sid += crossover_size
        for i in range(random_size):
            speciman = Speciman(self.params, self._sid + i)
            gen.append(speciman)
        self._sid += random_size
        for i in range(clones_size):
            speciman = Speciman(self.params, self._sid + i)
            for j in range(len(self.params)):
                speciman.values[j] = int(top[i % len(top)].values[j])
            gen.append(speciman)
        self._sid += clones_size
        self.specimen += gen
        self.generations += [gen]
        genid = self._gid
        self._gid += 1
        return (gen, genid)

    def mutate(self):
        for speciman in self.generations[-1]:
            for (idx, param) in enumerate(self.params):
                mult = self.mutation_ratio * param['mutation_mult'] # * random()
                if random() > mult:
                    continue
                oldv = speciman.values[idx]
                newv = randint(param['min'], param['max'])
                speciman.values[idx] = newv

    def mutate2(self):
        for speciman in self.generations[-1]:
            for (idx, param) in enumerate(self.params):
                mult = self.mutation_ratio * param['mutation_mult'] # * random()
                oldv = speciman.values[idx]
                d = param['max'] - param['min']
                newv = max(param['min'],
                           min(param['max'], oldv + d*mult*(1-2*random())))
                speciman.values[idx] = int(newv)

    def iterate(self):
        self.mutation_ratio *= self.mutation_decline
        if self._gid < 1:
            (gen, genid) = self.init_generation()
        else:
            (gen, genid) = self.crossover()
            if not genid % self.epoch:
                self.mutation_ratio = self.original_mutation_ratio
                for i in range(len(self.params)):
                    min_value = math.inf
                    max_value = -math.inf
                    for (score, id) in self.leaders:
                        v = self.specimen[id].values[i]
                        if v < min_value:
                            min_value = v
                        if v > max_value:
                            max_value = v
                    deltamin = min_value - self.params[i]['min']
                    deltamax = self.params[i]['max'] - max_value
                    # self.params[i]['min'] += deltamin/2
                    # self.params[i]['max'] -= deltamax/2
                    oldrng = self.params[i]['origmax'] - self.params[i]['origmin']
                    newrng = self.params[i]['max'] - self.params[i]['min']
                    if newrng/oldrng < (1 - self.hiscore):
                        center = self.params[i]['min'] + newrng/2
                        rng = oldrng * (1 - self.hiscore) / 2
                        self.params[i]['min'] = max(center - rng, self.params[i]['origmin'])
                        self.params[i]['max'] = min(center + rng, self.params[i]['origmax'])
                # print(self.params[0])
            self.mutate2()
        for speciman in gen:
            try:
                hash = speciman.hash()
                if hash in self.test_results:
                    continue
                else:
                    score = self.test_function(speciman)
                    self.test_results[hash] = score
            except:
                traceback.print_tb(sys.exc_info()[2])
                raise
            self.rate(speciman.id, score)
        return self.leaders[0][0]


if __name__ == '__main__':
    def test_function(speciman):
        (x, y, z) = speciman.values
        metric_volume = x * y * z / 4096
        if metric_volume > 1:
            metric_volume = 1/metric_volume
        metric_xy_ratio = x / y if y > 0 else 0
        if metric_xy_ratio > 1:
            metric_xy_ratio = 1/metric_xy_ratio
        metric_yz_ratio = y / z if z > 0 else 0
        if metric_yz_ratio > 1:
            metric_yz_ratio = 1/metric_yz_ratio
        return metric_volume * (0.5*metric_xy_ratio + 0.5*metric_yz_ratio)
    target = 0.9
    maxgens = 300
    gm = GeneticModel(pop_size=30, mutation_ratio=.2, epoch=24, mutation_decline=0.9)
    gm.add_param('x', 0, 100)
    gm.add_param('y', 0, 100)
    gm.add_param('z', 0, 100)
    gm.test_function = test_function
    for i in range(maxgens):
        hiscore = gm.iterate()
        if hiscore > target:
            print(f"Break at generation {i}")
            break
        # print(f'Run {i + 1}: {hiscore} @{gm.mutation_ratio}')
    print('=== Leaderboard ===')
    print('No.\tScore\tSpeciman values')
    i = 1
    for (score, spid) in gm.leaders:
        speciman = gm.specimen[spid]
        print(f'{i}.\t{"%.03f" % speciman.score}\t{speciman.values}')
        i += 1
    print('--- end ---')
