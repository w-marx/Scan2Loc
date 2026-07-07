import numpy as np

class UnionFind:
    def __init__(self, n:int):
        """
        :param n: the number of indices/max element indice to control, has to be > 0
        """
        assert n > 0, f"{n} is no valid size for a UnionFind structure"
        self.n = n
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)


    def find(self, e:int)->int:
        """
        Find operation with path compression
        :param e: the indice of which to return the representative idx
        :return the index of the representative
        """
        assert 0 <= e < self.n, f"Index {e} is not in UnionFind of size {self.n}"
        while self.parent[e] != e:
            self.parent[e] = self.parent[self.parent[e]]
            e = self.parent[e]
        return e


    def union(self, e1:int, e2:int):
        """
        Union by Rank between a & b, if they arent already in the same Cluster
        :param e1, the index of element 1 to merge
        :param e2, the index of the second element to merge
        """
        assert 0 <= e1 < self.n, f"Index {e1} is not in UnionFind of size {self.n}"
        assert 0 <= e2 < self.n, f"Index {e2} is not in UnionFind of size {self.n}"
        
        representative_e1 = self.find(e1)
        representative_e2 = self.find(e2)

        if representative_e1 == representative_e2:
            return

        if self.rank[representative_e1] < self.rank[representative_e2]:
            self.parent[representative_e1] = representative_e2
        elif self.rank[representative_e1] > self.rank[representative_e2]:
            self.parent[representative_e2] = representative_e1
        else:
            self.parent[representative_e2] = representative_e1
            self.rank[representative_e1] += 1
    

    def return_clusters(self)->list[list[int]]:
        """
        :return a list of all clusters (each cluster as a list of indices)
        """
        clusters = {}
        for idx, parent in enumerate(self.parent):
            representative = self.find(idx)
            if representative in clusters:
                clusters[representative].append(idx)
            else:
                clusters[representative] = [idx]
        return list(clusters.values())