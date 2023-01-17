import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
import copy
import itertools
import functools
from src.statistics import multinomial_coeff, gen_perm

def gen_valid_region(num=1):
    high = 0.5
    low = 0
    regions = []
    for i in range(num):
        left = round(uniform(low, high), 3)
        right = round(uniform(left, high), 3)
        regions.append((left, right))
        low = right
        high = 1
    return Region(regions)


class Region:

    def __init__(self, region_lst, remove_zero_measure=False):
        assert isinstance(region_lst, list)
        regions = [r for r in region_lst if r != ()]
        if len(regions) == 0:
            self.regions = [()]
            return
        for r in regions:
            assert isinstance(r, tuple)
            assert r[0] <= r[1]
        regions = sorted(regions)
        for r1, r2 in zip(regions[:-1], regions[1:]):
            assert r1[1] <= r2[0]

        self.regions = []
        if len(regions) > 1:
            regions += [()]
            merged = regions[0]
            last = regions[0]
            for r in regions[1 :]:
                if r == ():
                    self.regions.append(merged)
                    break
                elif last[1] == r[0]:
                    merged = (merged[0], r[1])
                else:
                    self.regions.append(merged)
                    merged = r
                last = r
        else:
            self.regions = regions
        if remove_zero_measure:
            self.regions = [r for r in self.regions if r[0] < r[1]]
            if len(self.regions) == 0:
                self.regions = [()]

    def __str__(self):
        return str(self.regions)

    def probability(self, normalizer=1):
        if self.regions == [()]:
            return 0
        s = 0
        for r in self.regions:
            s += r[1] - r[0]
        return s/normalizer

    @staticmethod
    def _intersection(r1, r2):
        assert isinstance(r1, tuple) and isinstance(r2, tuple)

        out = ()
        if r1 == () or r2 == ():
            return out

        if r1[0] <= r2[0] <= r1[1]:
            if r2[1] <= r1[1]:
                out = (r2[0], r2[1])
            else:
                out = (r2[0], r1[1])
        elif r2[0] <= r1[0] <= r2[1]:
            if r1[1] <= r2[1]:
                out = (r1[0], r1[1])
            else:
                out = (r1[0], r2[1])
        return out

    @staticmethod
    def _union(r1, r2):
        assert isinstance(r1, tuple) and isinstance(r2, tuple)

        if r1 == () or r2 == ():
            if r1 == () and r2 == ():
                out = [()]
            elif r1 == ():
                out = [r2]
            else:
                out = [r1]
        elif r1[0] < r2[0] < r1[1]:
            if r2[1] < r1[1]:
                out = [r1]
            else:
                out = [(r1[0], r2[1])]
        elif r2[0] < r1[0] < r2[1]:
            if r1[1] < r2[1]:
                out = [r2]
            else:
                out = [(r2[0], r1[1])]
        else:
            out = sorted([r1, r2])

        return out

    def contain(self, x):
        if isinstance(x, Region):
            for r in x.regions:
                res = [a for a in self.regions if a[0] <= r[0] <= r[1] <= a[1]]
                if len(res) == 0:
                    return False
            return True
        elif isinstance(x, tuple):
            res = [a for a in self.regions if a[0] <= x[0] <= x[1] <= a[1]]
            if len(res) == 0:
                return False
            else:
                return True
        else:
            res = [a for a in self.regions if a[0] <= x <= a[1]]
            if len(res) == 0:
                return False
            else:
                return True


    def intersection(self, region_obj):
        if self.regions == [()] or region_obj.regions == [()]:
            return Region([()])
        sections = []
        for r1 in self.regions:
            for r2 in region_obj.regions:
                intersected = Region._intersection(r1, r2)
                if intersected != ():
                    sections.append(intersected)
        region_intersected = Region(sections)
        return region_intersected

    def union(self, region_obj):
        intersect = self.intersection(region_obj)
        subt1 = self._subtract(region_obj)
        subt2 = region_obj._subtract(self)
        union_lst = intersect.regions + subt1 + subt2
        return Region(union_lst)

    def subtract(self, region_obj):
        return Region(self._subtract(region_obj), remove_zero_measure=True)

    def _subtract(self, region_obj):
        if self.regions == [()]:
            return [()]
        elif region_obj.regions == [()]:
            return copy.deepcopy(self.regions)
        subt_lst = []
        r1_iter = iter(self.regions + [()])
        r2_iter = iter(region_obj.regions + [()])
        r1 = next(r1_iter)
        r2 = next(r2_iter)
        while True:
            if r1 == ():
                break
            elif r2 == ():
                subt_lst.append(r1)
                r1 = next(r1_iter)
                while r1 != ():
                    subt_lst.append(r1)
                    r1 = next(r1_iter)
                break
            elif r1[0] <= r2[0] <= r2[1] <= r1[1]:
                subt_lst.append((r1[0], r2[0]))
                r1 = (r2[1], r1[1])
                r2 = next(r2_iter)
            elif r1[0] <= r2[0] <= r1[1]:
                subt_lst.append((r1[0], r2[0]))
                r2 = (r1[1], r2[1])
                r1 = next(r1_iter)
            elif r2[1] <= r1[0]:
                r2 = next(r2_iter)
            elif r2[0] <= r1[0] <= r1[1] <= r2[1]:
                r2 = (r1[1], r2[1])
                r1 = next(r1_iter)
            elif r2[0] <= r1[0] <= r2[1]:
                r1 = (r2[1], r1[1])
                r2 = next(r2_iter)
            elif r1[1] <= r2[0]:
                subt_lst.append(r1)
                r1 = next(r1_iter)
        return subt_lst


class SecondTerm:
    def __init__(self, D, Dp, X=Region([(0,1)])):
        assert len(D) == len(Dp)
        self.D = D
        self.Dp = Dp
        self.Dc = X.subtract(functools.reduce(lambda a, b: a.union(b), D+Dp, Region([()])))
        self.D_separated = SecondTerm.get_separate_regions(D)
        self.Dp_separated = SecondTerm.get_separate_regions(Dp)
        self.D_separated_dict = {}
        self.Dp_separated_dict = {}
        for i in range(len(self.D_separated)):
            self.D_separated_dict[i] = [j for j in range(len(D)) if D[j].contain(self.D_separated[i])]
        for i in range(len(self.Dp_separated)):
            self.Dp_separated_dict[i] = [j for j in range(len(Dp)) if Dp[j].contain(self.Dp_separated[i])]
        self.D_separated_prob = [r.probability() for r in self.D_separated]
        self.Dp_separated_prob = [r.probability() for r in self.Dp_separated]
        self.Dc_prob = self.Dc.probability()

    @staticmethod
    def get_separate_regions(regions):
        separated = []
        k = len(regions)
        k_range = list(range(k))
        for r in range(1, k+1):
            combinations = itertools.combinations(k_range, r)
            for comb in combinations:
                complement = [i for i in k_range if i not in comb]
                intersection_lst = [regions[i] for i in comb]
                union_lst = [regions[i] for i in complement]
                intersect = functools.reduce(lambda a, b: a.intersection(b), intersection_lst)
                union = functools.reduce(lambda a, b: a.union(b), union_lst, Region([()]))
                subt = intersect.subtract(union)
                separated.append(subt)
        separated = [r for r in separated if r.regions != [()]]
        return separated

    def probability(self, n, ell=0):
        k = len(self.D)
        m1 = len(self.D_separated)
        m2 = len(self.Dp_separated)
        m = m1 + m2 + 1 if self.Dc_prob > 0 else m1 + m2

        def validate_perm(perm1, perm2, D_separated_dict, Dp_separated_dict, k, ell=0):
            m1 = len(perm1)
            m2 = len(perm2)
            D_lst = [0]*k
            Dp_lst = [0]*k
            for i in range(k):
                D_lst[i] = sum([perm1[j] for j in range(m1) if i in D_separated_dict[j]])
                Dp_lst[i] = sum([perm2[j] for j in range(m2) if i in Dp_separated_dict[j]])
                if D_lst[i] + ell <= Dp_lst[i]:
                    return False
            return zip(D_lst, Dp_lst)

        cumsum = 0
        for perm in gen_perm(n, m):
            perm1 = perm[:m1]
            perm2 = perm[m1:m1+m2]

            if validate_perm(perm1, perm2, self.D_separated_dict, self.Dp_separated_dict, k, ell):
                cumsum += multinomial_coeff(n, perm) * np.prod(np.array(self.D_separated_prob + self.Dp_separated_prob + [self.Dc_prob]) ** np.array(perm))
        return cumsum








if __name__ == '__main__':
    #D = [Region([(0.6, 0.9)]), Region([(0.4, 0.6)])]
    #Dp = [Region([(0.9, 1)]), Region([(0.3, 0.4)])]
    D = [Region([(0.6, 0.9)])]
    Dp = [Region([(0.9, 1)])]

    a = SecondTerm(D, Dp)

    n_range = list(range(2, 302, 10))

    exponent_arr = np.array([])
    for n in n_range:
        p = 1 - a.probability(n)
        exponent = -(1/n)*np.log(p)
        exponent_arr = np.append(exponent_arr, exponent)
        print(exponent_arr[-1])
    plt.figure()
    plt.plot(n_range, exponent_arr)
    plt.axhline(y=0.0551, color='r', linestyle='-')
    plt.ylim([0, 0.32])
    plt.xlabel('n')
    plt.ylabel('probability')
    plt.legend([r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D^f_j \leq $#$D$'#$^f_j$}$_n)$", '$exp(-nD(\Pi^t || Q^t))$'])
    plt.title('teacher first order exponent')
    plt.show()









