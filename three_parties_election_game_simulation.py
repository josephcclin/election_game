#!/usr/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 9 2022

Dealing with election game for three or more parties
Special thanks for Roman Akchurin for his assistance 
(for the two-party election game instance generation)

Joseph Chuang-Chieh Lin

"""
import numpy as np


class Natural(object):
    """The natural model for computing the winning prob. for a party."""
    @staticmethod
    def fn(A, B, C, social_bound):
        """Takes `A`, `B`, `C` as the utility matrices for party A, B, C, resp. 
        social_bound: the upper bound assumption on the total social utility.

        Returns `P` the probabilities of winning of a candidate `i` of party A
        against the competing candidate `j` of party B and candidate `k`of 
        party C as three-dimensional Numpy tensor according to the natural model."""
        l = A.shape[0] # number of candidates in party A
        m = B.shape[0] # number of candidates in party B
        n = C.shape[0] # number of candidates in party C
        
        P = np.zeros((l,m,n,3))
        for i in range(l):
            for j in range(m):
                for k in range(n):
                    uA_i = A[i,:].sum()
                    uB_j = B[j,:].sum()
                    uC_k = C[k,:].sum()
                    P[i,j,k] = np.array([uA_i/(uA_i+uB_j+uC_k), 
                                         uB_j/(uA_i+uB_j+uC_k),
                                         uC_k/(uA_i+uB_j+uC_k)])
        return P

class Softmax(object):
    """Softmax model for computing the winning odds for a party."""
    @staticmethod
    def fn(A, B, C, social_bound):
        """Takes `A`, `B`, `C` as the utility matrices for party A, B, C, resp. 
        social_bound: the upper bound assumption on the total social utility.

        Return `P` the probabilities of winning of a candidate `i` of party A 
        against the competing candidate `j` of party B and candidate `k` of 
        party C as three-dimensional Numpy tensor according to the softmax model."""
        l = A.shape[0] # number of candidates in party A
        m = B.shape[0] # number of candidates in party B
        n = C.shape[0] # number of candidates in party C
        
        P = np.zeros((l,m,n,3))
        for i in range(l):
            for j in range(m):
                for k in range(n):
                    uA_i = A[i,:].sum() / social_bound
                    uB_j = B[j,:].sum() / social_bound
                    uC_k = C[k,:].sum() / social_bound
                    P[i,j,k] = np.array([np.exp(uA_i)/(np.exp(uA_i) + np.exp(uB_j) + np.exp(uC_k)), 
                                         np.exp(uB_j)/(np.exp(uA_i) + np.exp(uB_j) + np.exp(uC_k)), 
                                         np.exp(uC_k)/(np.exp(uA_i) + np.exp(uB_j) + np.exp(uC_k))])
        return P

class ElectionGame(object):
    """Election game object implementation."""
    def __init__(self, num_candidates=(2,2,2), social_bound=100, model=Softmax, \
        force_egoism=False, seed=None):
        self.num_candidates = num_candidates
        self.social_bound = social_bound
        self.model = model
        self.force_egoism = force_egoism
        self.rng = np.random.default_rng(seed)
        self.history = list()
        self.NoPNE = list()

    def generate_parties(self):
        """Generate a party with the following properties:
        `self.social_bound` is the bound of social utilities;
        `self.num_candidates` is the numbers of candidates in the parties.
        
        Supporters for the candidate could be:
            (a) supporters for the candidate's party;
            (b) supporters for the opposing party;
            
        Return the generated party utility matrix as three-dimensional Numpy tensor."""
        voter_types = 3 
        # create the first dummy candidate
        parties = []
        for g in range(3):
            party = np.zeros((1,voter_types), dtype=np.int64)
            for i in range(self.num_candidates[g]):
                while True:
                    # create candidates one by one
                    candidate = np.array([])
                    UB = self.social_bound
                    for a in range(voter_types):
                        new_value = self.rng.integers(0, UB*100, dtype=np.int64)
                        candidate = np.append(candidate, new_value/100)
                        UB = (UB*100 - new_value)/100
                    #candidate = self.rng.integers(0, self.social_bound, \
                    #    size=(1,voter_types), dtype=np.int64, endpoint=True)
                    candidate = candidate.reshape(1,voter_types)
                    if np.sum(candidate.sum(axis=1) <= self.social_bound) == 1:
                        break
                party = np.vstack((party,candidate))
            # remove the first dummy candidate
            party = party[1:,:]
            # sort based on the number of supporters for the candidates's party
            #party = party[np.argsort(np.array([party[:,g] for i in range(3)]))][::-1]
            party = party[np.argsort(party[:, g])][::-1]
            parties.append(party)
        return parties

    def confirm_egoism(self, A, B, C):
        """Test whether the game is egoistic.

        Takes `A`, `B`, `C` as the utility matrices for parties `A`, `B`, `C`, resp. 
        Return a boolean value True or False."""
        for i in range(A.shape[0]):
            #UT_ALL = A[0,0] + A[0,1] + A[0,2] 
            if A[i,0] <= max(B[:,0].max(), C[:,0].max()): #B[:,0].max()+C[:,0].max():#
                return False
            #if i>0 and UT_ALL < A[i,0]+A[i,1]+A[i,2]:
            #    return False
        for i in range(B.shape[0]):
            #UT_ALL = B[0,0] + B[0,1] + B[0,2]
            if B[i,1] <= max(A[:,1].max(), C[:,1].max()): #A[:,1].max()+C[:,1].max():#
                return False
            #if i>0 and UT_ALL < B[i,0]+B[i,1]+B[i,2]:
            #    return False
        for i in range(C.shape[0]):
            #UT_ALL = C[0,0] + C[0,1] + C[0,2]
            if C[i,2] <= max(A[:,2].max(), B[:,2].max()): #A[:,2].max()+B[:,2].max():#
                return False
            #if i>0 and UT_ALL < C[i,0]+C[i,1]+C[i,2]:
            #    return False
        return True

    def get_payoffs(self, A, B, C, P):
        """Compute the payoffs of a two-party election game as expected
        utilities.
        
        Takes `A`, `B`, and `C` as the utility matrices for parties `A`, `B`, `C`, resp. 
        `P`: the probabilities of winning according to the selected model.
        
        Returns `a`, `b`, `c` as the payoffs of party `A`, `B`, `C` resp., as
        three-dimensional Numpy tensors."""
        a = np.zeros(self.num_candidates)#np.zeros((self.num_candidates, self.num_candidates, self.num_candidates))
        b = np.zeros(self.num_candidates)#np.zeros((self.num_candidates, self.num_candidates, self.num_candidates))
        c = np.zeros(self.num_candidates)#np.zeros((self.num_candidates, self.num_candidates, self.num_candidates))
        
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                for k in range(C.shape[0]):
                    a[i,j,k] = P[i,j,k][0]*A[i,0] + P[i,j,k][1]*B[j,0] + P[i,j,k][2]*C[k,0]
                    b[i,j,k] = P[i,j,k][0]*A[i,1] + P[i,j,k][1]*B[j,1] + P[i,j,k][2]*C[k,1]
                    c[i,j,k] = P[i,j,k][0]*A[i,2] + P[i,j,k][1]*B[j,2] + P[i,j,k][2]*C[k,2]
        return (a,b,c)

    def get_optimal_state(self, a, b, c, social_welfare):
        """Computes the optimal state, which has the highest social welfare
        among all possible states.

        Take payoff `a` of party A, payoff `b` of party B, and 
        payoff `c` of party C.

        Returns the best social welfare value as a float."""
        max_idx = np.unravel_index(np.argmax(social_welfare, axis=None), \
            social_welfare.shape)
        return social_welfare[max_idx]

    def get_worst_PNE(self, a, b, c, social_welfare):
        """Computes Pure Nash Equilibrium.

        Take payoff `a`, `b`, `c` of party A, B, C resp. 
        Return position as a tuple and worst pure-strategy Nash equilibrium 
        as a float."""
        PNEs = list()

        for i in range(self.num_candidates[0]):
            for j in range(self.num_candidates[1]):
                for k in range(self.num_candidates[2]):
                    if a[i,j,k] == a[:,j,k].max() and b[i,j,k] == b[i,:,k].max() and c[i,j,k] == c[i,j,:].max():
                        PNEs.append(((i,j,k), social_welfare[i,j,k]))
        # sort PNEs in ascending order
        PNEs.sort(key=lambda a: a[1])
        # return the worst (smallest in value) PNE
        if len(PNEs) > 0:
            return PNEs[0]
        else:
            return ((None, None, None), None)

    def get_PoA(self, optimal_val, PNE_val):
        """Calculates the price of anarchy.

        Takes `optimal_state` the highest social welfare and `PNE_val` the
        value of the worst pure Nash equilibrium.

        Returns the Price of Anarchy as a float."""
        if PNE_val == None or PNE_val == 0:
            return 0
        else:
            return optimal_val / PNE_val

    def run_election(self):
        """Runs the election process once.

        Return `A`, `B`, `C` as the utility matrices for party `A`, `B`, `C`, resp.  
        Return `a`, `b`, `c` as the payoffs of party A, B, C, resp. 
        `PNE_pos`: the position of worst pure-strategy Nash equilibrium, 
        `PNE_val`: the value of worst pure-strategy Nash equilibrium, 
        `PoA`: the Price of Anarchy,
        all packed into a single tuple for further recording in the history."""
        parties = self.generate_parties()
        A, B, C = parties[0], parties[1], parties[2]

        if self.force_egoism:
            while not self.confirm_egoism(A, B, C):
                parties = self.generate_parties()
                A, B, C = parties[0], parties[1], parties[2]

        P = self.model.fn(A, B, C, self.social_bound)
        a, b, c = self.get_payoffs(A, B, C, P)
        social_welfare = a + b + c
        optimal_val = self.get_optimal_state(a, b, c, social_welfare)
        (PNE_pos, PNE_val) = self.get_worst_PNE(a, b, c, social_welfare)
        PoA = self.get_PoA(optimal_val, PNE_val)
        return (A, B, C, P, a, b, c, optimal_val, PNE_pos, PNE_val, PoA)

    def run_iterations(self, iterations):
        """Runs election `iterations` times. The history is stored in
        `self.history` class variable for further analysis."""
        for i in range(iterations):
            (A, B, C, P, a, b, c, optimal_val, PNE_pos, PNE_val, PoA) = self.run_election()
            self.history.append((A, B, C, P, a, b, c, optimal_val, PNE_pos, PNE_val, PoA))
            if PNE_val == None:
                print("*** NoPNE has been found: *** ")
                print(A, B, C, P, a, b, c, optimal_val, PNE_pos, PNE_val, PoA)
                self.NoPNE.append((A, B, C, P, a, b, c, optimal_val, PNE_pos, PNE_val, PoA))
            
        worst_PoA = max([record[-1] for record in self.history])
        n_PNEs = len([record[-2] for record in self.history if record[-2]])
        n_records = len(self.history)
        print(f'Model: {self.model.__name__}')
        print(f'Worst PoA: {worst_PoA:.2f}')
        print(f'Found PNE: {n_PNEs}/{n_records}')

if __name__ == "__main__":
    polgame = ElectionGame(num_candidates=(2,2,2), social_bound=100, \
        model=Softmax, force_egoism=True, seed=None)
    polgame.run_iterations(5000)
