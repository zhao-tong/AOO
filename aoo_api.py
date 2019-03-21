import sys
import copy
import time
import getopt
import random
import fraudar
import numpy as np
import matplotlib.pylab as plt
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, load_npz, save_npz
from numba import vectorize, float64, int32, jit


# This is a logistic function:
#   g(x) = 1/(1+e^(alpha*x)).
# When x > 6, g(x) > 0.997 and we return 1.
# When x < -6, g(x) < 0.003 and we return 0.
@vectorize(['float64(float64, float64, int32)'], target='parallel')
def g(x, m, alpha):
    alpha_x = alpha * (x - m)
    if alpha_x > 6: return 1.0
    if alpha_x < -6: return 0.0
    return 1.0/(1.0+np.exp(-alpha_x))

# The function that generates the ratings.
@vectorize(['float64(float64, float64)'], target='parallel')
def generate(I, p):
    return I * (int(np.random.rand()**p * 5)/4)

class AOO_API:

    def __init__(self, _n, _m, n0, m0, da, db, dc, pa, ngroup, read, v_init=0.1, c_init=0.1, nstage=30):
        self._n = _n
        self._m = _m
        self.n0 = n0
        self.m0 = m0
        self.da = da
        self.db = db
        self.dc = dc
        self.pa = pa
        self.ngroup = ngroup
        self.n = _n + ngroup * n0
        self.m = _m
        self.alpha = 1000
        self.go = True
        self.v = None
        self.beta_u = 1
        self.beta_v = 1
        self.A = None
        self.I = None
        self.u = None
        self.v_init = v_init
        self.c_init = c_init
        self.nstage = nstage
        if read != None:
            [self.A, self.I, self.u] = self.readMatrix(read)
        else:
            [self.A, self.I, self.u] = self.initMatrix()
        (self.n, self.m) = np.shape(self.I)
        self.original_I = copy.deepcopy(self.I)
        self.original_u = copy.deepcopy(self.u)
        print(np.shape(self.A), np.shape(self.I), np.shape(self.u))

    # The function that initiates the data matrices by given settings.
    def initMatrix(self):
        # generate A (n*m): rating matrix
        # generate I (n*m): indicator matrix
        Ia = np.array([1]*self.da*self._n + [0]*(self._n*self.m - self.da*self._n))
        np.random.shuffle(Ia)
        Ia = Ia.reshape(self._n, self.m)

        jlen = self.m - self.ngroup * self.m0
        Ib = np.array([1]*self.db*self.ngroup*self.n0 + [0]*(jlen*self.ngroup*self.n0 - self.db*self.ngroup*self.n0))
        np.random.shuffle(Ib)
        Ib = Ib.reshape(self.ngroup*self.n0, jlen)

        Ic = np.zeros((self.ngroup*self.n0, self.ngroup*self.m0))
        Ibc = np.concatenate((Ib, Ic), axis=1)
        I = np.concatenate((Ia, Ibc), axis=0)

        A = generate(I, self.pa)

        for k in range(self.ngroup):
            for i in range(self.n0):
                arr = np.array([1]*self.dc + [0]*(self.m0-self.dc))
                np.random.shuffle(arr)
                for j in range(self.m0):
                    if arr[j] == 1:
                        ii = i + self._n + k * self.n0
                        jj = j + (self.m - (k + 1) * self.m0)
                        I[ii, jj] = 1
                        A[ii, jj] = self.c_init * random.random()

        # compute u (n*1): average rating
        sumu = np.sum(A,1)
        numu = np.sum(I,1)
        u = np.divide(sumu, numu)
        u = np.nan_to_num(u)
        u = np.mat(u).T

        plt.imshow(I,interpolation='none',cmap='Greys')
        plt.colorbar()
        plt.show()

        return [A, I, u]

    def readMatrix(self, name):
        filename_A = 'A{}.npz'.format(name)
        filename_I = 'I{}.npz'.format(name)
        sA = load_npz(filename_A)
        sI = load_npz(filename_I)
        (n, m) = np.shape(sA)
        self.n = n
        self.m = m
        A = np.array(sA.todense())
        I = np.array(sI.todense())
        sumu = np.sum(A,1)
        numu = np.sum(I,1)
        u = np.divide(sumu, numu)
        u = np.nan_to_num(u)
        u = np.mat(u).T
        return [A, I, u]

    def save_I(self):
        sA = csr_matrix(self.A)
        sI = csr_matrix(self.original_I)
        save_npz('Ang.npz', sA)
        save_npz('Ing.npz', sI)
        fw = open('data/socialgraph', 'w')
        for i in range(self.n):
            for j in range(self.m):
                if self.original_I[i, j] != 0:
                    fw.write('{}\t{}\n'.format(i, j+3000))
        fw.close()

    # The function that gets the value of J and some temp values that will be used later.
    def getJ(self, criteria):
        # calc B
        bv = np.mat(np.ones(self.n)).T @ self.v.T
        bu = self.u @ np.mat(np.ones(self.m))
        B = bv - bu
        B = g(B, 0, self.alpha)
        B = np.multiply(self.I, B)

        # compute cu (n*1): number of u's ratings being blocked
        cu = np.sum(B,1)
        # compute cv (m*1): number of ratings v blocks
        cv = np.sum(B,0).T
        #print(np.sum(B, 0))

        # compute su (n*1): if u's #block exceeds beta_u -> in the subgraph
        su = g(cu, self.beta_u, self.alpha)
        # compute sv (m*1): if v's #block exceeds beta_v -> in the subgraph
        sv = g(cv, self.beta_v, self.alpha)
        #print(sv)
        # compute nu (1*1): number of blocked u in the subgraph
        nu = np.sum(su)
        # compute nv (1*1): number of blocked v in the subgraph
        nv = np.sum(sv)

        if nu == 0 or nv == 0:
            self.go = False

        # compute ne (1*1): number of blocked ratings in the subgraph
        ne = su.T @ B @ sv
        ne = np.asscalar(ne)

        # compute J: objective function
        J = 0.0
        if nu > 0 and nv > 0:
            if criteria == 'sing':
                J = ne/np.sqrt(nu*nv)
            else:
                J = ne/nu/nv

        return [J, B, su, sv, nu, nv, ne]

    # The function that gets the value of J and the derivative of J respecting v.
    def calcJDJ(self, criteria):
        #print(1)
        [J, B, su, sv, nu, nv, ne] = self.getJ(criteria)
        su1su = np.multiply(su, 1-su)
        sv1sv = np.multiply(sv, 1-sv)

        B1g = np.multiply(B, 1-B)

        d_nu_dv = (su1su.T @ B1g) * self.alpha * self.alpha
        d_nv_dv = np.multiply(np.sum(B1g, 0), sv1sv.T) * self.alpha * self.alpha
        Bsu = B.T @ su
        Bsv = B @ sv

        d_ne_dv0 = np.asscalar((su1su.T @ Bsv) * self.alpha)
        d_ne_dv2 = sv.T * nu
        d_ne_dv1 = np.multiply(sv1sv.T, Bsu.T * self.n) * self.n * self.alpha
        d_ne_dv = d_ne_dv1 + d_ne_dv2 + d_ne_dv0
        d_J_dv = np.mat(np.zeros(self.m))
        if nu > 0 and nv > 0:
            d_J_dv = (d_ne_dv / (nu*nv)) - (d_nu_dv*ne / (nu*nu*nv)) - (d_nv_dv*ne / (nu*nv*nv))

        return [J, d_J_dv]

    # The function that updates I and u by deleting the data of buyers in su from I.
    def update(self, v_now):
        # update I
        # new_I[bv > bu] = 0?
        new_I = copy.deepcopy(self.I)

        '''
        bv = np.mat(np.ones(self.n)).T @ self.v.T
        bu = self.u @ np.mat(np.ones(self.m))
        B = bv - bu
        B = g(B, 0, self.alpha)
        B = np.multiply(self.I, B)
        # compute cu (n*1): number of u's ratings being blocked
        cu = np.sum(B,1)
        for i in range(self.n):
            if cu[i] >= self.beta_u:
                new_I[i] = np.zeros(self.m)
        '''
        for i in range(self.n):
            for j in range(self.m):
                if new_I[i, j] == 1 and v_now[j] > self.u[i]:
                    new_I[i, j] = 0

        # update u
        sumu = np.sum(self.A, 1)
        numu = np.sum(new_I, 1)
        new_u = np.divide(sumu, numu)
        new_u = np.nan_to_num(new_u)
        new_u = np.mat(new_u).T
        self.I = copy.deepcopy(new_I)
        self.u = copy.deepcopy(new_u)

    # The function that gets the best v base on all history versions of v.
    #   it picks the largest for each seller.
    def getBestv(self, v_history):
        final_v = np.zeros(self.m)
        for i in range(self.m):
            vi = 0
            for hist_v in v_history:
                if hist_v[i] > vi:
                    vi = hist_v[i]
            final_v[i] = vi
        return final_v

    def combineV(self, last_v, current_v):
        if last_v.all() == 0:
            return current_v
        else:
            new_v = copy.deepcopy(current_v)
            for i in range(len(current_v)):
                if last_v[i] > current_v[i]:
                    new_v[i] = last_v[i]
            return new_v

    # This function evaluates the performance (precision, recall, f1),
    #   assuming the ground truth is, i in [n-n0,n).
    def evaluate(self, u, v, I):
        #caught = list()
        precision,recall,f1,acc = 0.0, 0.0, 0.0, 0.0
        hits,predict,truth = 0, 0, 0
        for i in range(self.n):
            for j in range(self.m):
                if I[i,j] == 1:
                    if u[i] < v[j]:
                        predict += 1
                    if i >= self._n:
                        truth += 1
                        if u[i] < v[j]:
                            hits += 1
        if predict > 0:
            precision = hits / predict
        if truth > 0:
            recall = hits / truth
        if precision > 0 and recall > 0:
            f1 = 2*precision*recall/(precision+recall)
        tot = np.sum(I)
        TN = tot - predict - truth + hits
        if tot > 0:
            acc = (hits + TN) / tot
        return [predict, truth, acc, precision, recall, f1]

    def evaluate_fraudar(self, pred):
        for i in range(len(pred)):
            pred[i] = int(pred[i])

        precision,recall,f1,acc = 0.0, 0.0, 0.0, 0.0
        hits,predict,truth = 0, 0, 0
        for i in range(self.n):
            for j in range(self.m):
                if self.original_I[i,j] == 1:
                    if i in pred:
                        predict += 1
                    if i >= self._n:
                        truth += 1
                        if i in pred:
                            hits += 1
        if predict > 0:
            precision = hits / predict
        if truth > 0:
            recall = hits / truth
        if precision > 0 and recall > 0:
            f1 = 2*precision*recall/(precision+recall)
        tot = np.sum(self.original_I)
        TN = tot - predict - truth + hits
        if tot > 0:
            acc = (hits + TN) / tot
        return [predict, truth, acc, precision, recall, f1]

    def fraudar(self):
        '''
        plt.imshow(self.original_I,interpolation='none',cmap='Greys')
        plt.colorbar()
        plt.show()
        '''
        pred = list()
        sparse_I = csr_matrix(self.original_I)
        reviews = [(i, j) for i, j in zip(*sparse_I.nonzero())]

        graph = fraudar.ReviewGraph(1, fraudar.aveDegree)

        # Create reviewers and products.
        reviewers = [graph.new_reviewer("{0}".format(i)) for i in range(self.n)]
        products = [graph.new_product("{0}".format(i)) for i in range(self.m)]

        # Add reviews.
        for review in reviews:
            graph.add_review(reviewers[review[0]], products[review[1]], 1)

        # Run the algorithm.
        graph.update()

        # write anomalous reviewrs to output.txt
        for r in graph.reviewers:
            if r.anomalous_score == 1:
                pred.append(r.name)

        _eval = self.evaluate_fraudar(pred)
        print('Fraudar: ', _eval)
        return pred, _eval

    def fraudar_v(self):
        pred, f_eval = self.fraudar()
        f_v = np.mat(np.zeros(self.m)).T
        bu = self.original_u @ np.mat(np.ones(self.m))
        bu = np.multiply(self.original_I, bu)
        for j in range(self.m):
            rates = dict()
            for i in range(self.n):
                if bu[i, j] != 0:
                    if i in pred:
                        rates[bu[i, j]] = 1
                    else:
                        rates[bu[i, j]] = 0
            f_v[j] = self.get_cut(rates)
        _eval = self.evaluate(self.original_u, f_v, self.original_I)
        print('Fraudar_v:', _eval)
        return f_eval, _eval

    def catchsync(self):
        pred = list()
        with open('data/unorcohoutdP2', 'r') as fr:
            for line in fr:
                arr = line.strip('\r\n').split(',')
                if float(arr[2]) > 0.4:
                    pred.append(int(arr[0]))
        c_eval = self.evaluate_fraudar(pred)
        #actionable catchsync
        c_v = np.mat(np.zeros(self.m)).T
        bu = self.original_u @ np.mat(np.ones(self.m))
        bu = np.multiply(self.original_I, bu)
        for j in range(self.m):
            rates = dict()
            for i in range(self.n):
                if bu[i, j] != 0:
                    if i in pred:
                        rates[bu[i, j]] = 1
                    else:
                        rates[bu[i, j]] = 0
            c_v[j] = self.get_cut(rates)
        ac_eval = self.evaluate(self.original_u, c_v, self.original_I)
        return c_eval, ac_eval

    # A support function that serves for fraudar_v, this function finds the 
    # best threshold that optimizes the accuracy.
    def get_cut(self, rates):
        s = sorted(rates.keys())
        cut, acc = 0.0, 0.0
        for i in range(len(s)-1):
            c = np.mean([s[i], s[i+1]])
            count = 0
            for key in rates:
                if key < c:
                    if rates[key] == 1:
                        count += 1
                else:
                    if rates[key] == 0:
                        count += 1
            a = count / len(s)
            if a > acc:
                cut = c
                acc = a
        return cut

    def spoken(self):
        pred = list()
        total_n = self.n + self.m
        Am = np.zeros((total_n, total_n))
        for i in range(self.n):
            for j in range(self.m):
                if self.original_I[i, j] == 1:
                    Am[i, j+self.n] = 1
                    Am[j+self.n, i] = 1
        e, v = eigh(Am, eigvals=(total_n-10, total_n-1))
        for i in range(10):
            e_vector = np.abs(v[:, -1-i])
            kmeans = KMeans(n_clusters=2, random_state=0).fit(e_vector.reshape(-1, 1))
            list_0 = list()
            list_1 = list()
            label = None
            for x in range(len(kmeans.labels_)):
                if kmeans.labels_[x] == 0:
                    list_0.append(e_vector[x])
                else:
                    list_1.append(e_vector[x])
            if np.mean(list_1) > np.mean(list_0):
                label = 1
            else:
                label = 0
            for x in range(len(kmeans.labels_)):
                if kmeans.labels_[x] == label and x < self.n:
                    pred.append(x)
        pred = list(set(pred))
        s_eval = self.evaluate_fraudar(pred)
        
        #actionable spoken
        s_v = np.mat(np.zeros(self.m)).T
        bu = self.original_u @ np.mat(np.ones(self.m))
        bu = np.multiply(self.original_I, bu)
        for j in range(self.m):
            rates = dict()
            for i in range(self.n):
                if bu[i, j] != 0:
                    if i in pred:
                        rates[bu[i, j]] = 1
                    else:
                        rates[bu[i, j]] = 0
            s_v[j] = self.get_cut(rates)
        as_eval = self.evaluate(self.original_u, s_v, self.original_I)
        print(s_eval, as_eval)
        return s_eval, as_eval

    def converge(self):
        np.seterr(divide='ignore', invalid='ignore')
        criteria = 'dens' # 'dens' (default) or 'sing'
        if criteria == 'sing': # learning rate: 0.025 (default)
            learning = 0.0001
        else:
            learning = 0.01#0.0025
        ntrial = 5 # number of trials

        v_history = list()
        count = 0
        stage = 0
        result = None
        while True:
            five_v = [None] * 5
            bestTrail = 0
            bestEval = None
            current_v = None
            if count == self.nstage:
                break

            bestJ = -1
            # optimization
            for _trial in range(ntrial):

                # initialize v (m*1): blocklist threshold
                self.v = np.mat(self.v_init * np.random.rand(self.m)).T

                J,lastJ,lastv,cJ = -1,-1,None,0
                cc = 0
                while True:
                    cc += 1
                    if cc > 50:
                        break
                    # compute derivatives
                    [J, d_J_dv] = self.calcJDJ(criteria)

                    if not self.go: break

                    if lastJ < 0:
                        lastJ = J
                        lastv = copy.deepcopy(self.v)
                        cJ = 1
                    elif J == lastJ:
                        cJ += 1
                        if cJ == 2:
                            break
                    elif J < lastJ:
                        J = lastJ
                        self.v = copy.deepcopy(lastv)
                        break
                    else:
                        lastJ = J
                        lastv = copy.deepcopy(self.v)
                        cJ = 1

                    # maximium 1.0
                    self.v += learning * d_J_dv.T
                    for j in range(0, self.m):
                        self.v[j] = max(0.0, min(1.0, self.v[j]))

                five_v[_trial] = self.v
                _eval = self.evaluate(self.u, self.v, self.I)
                #print('|stage: {3}, trail: {0}, final\t{1}\t{2}'.format(_trial, J, _eval, count+1))
                if J > bestJ:
                    bestJ = J
                    bestEval = _eval
                    bestTrail = _trial

            current_v = five_v[bestTrail]
            v_history.append(current_v)
            v_now = self.getBestv(v_history)
            _eval = self.evaluate(self.original_u, v_now, self.original_I)
            print('|stage: {0}'.format(count+1))
            print('|best trial: {0}, eval: {1}'.format(bestTrail, bestEval))
            print('|current comb eval: {0}'.format(_eval))
            print('|---------------------------')

            if stage == 0 and _eval[-1] == 1:
                stage = count + 1
                result = _eval
                break
            if count == 9:
                result = _eval

            # delete blocked submatrix
            self.update(v_now)

            count += 1

        if stage == 0:
            stage = 40
        final_v = self.getBestv(v_history)
        _eval = self.evaluate(self.original_u, final_v, self.original_I)
        print('|_n={0}, _m={1}, n0={2}, m0={3}, da={4}, db={5}, dc={6}, pa={7}, \
            ngroup={8}'.format(self._n, self._m, self.n0, self.m0, self.da, 
                self.db, self.dc, self.pa, self.ngroup))
        print('|final eval: {0}'.format(_eval))
        return stage, result

if __name__ == '__main__':
    _n = 2000
    _m = 2000
    n0 = 100
    m0 = 100
    da = 20
    db = 0
    dc = 20
    pa = 0.25
    ngroup = 3
    read = None
    try:
        args = sys.argv[1:]
        opts, args = getopt.getopt(args, '', ['_n=', '_m=', 'n0=', 'm0=', 'da=', 'db=',\
            'dc=', 'pa=', 'ngroup=', 'read'])
        print('Inputs:', opts)
        for opt, arg in opts:
            if opt == '--_n':
                _n = int(arg)
            elif opt == '--_m':
                _m = int(arg)
            elif opt == '--n0':
                n0 = int(arg)
            elif opt == '--m0':
                m0 = int(arg)
            elif opt == '--da':
                da = int(arg)
            elif opt == '--db':
                db = int(arg)
            elif opt == '--dc':
                dc = int(arg)
            elif opt == '--pa':
                pa = float(arg)
            elif opt == '--ngroup':
                ngroup = int(arg)
            elif opt == '--read':
                read = arg
    except Exception:
        print('argument parser error: {}'.format(Exception.args))
    finally:
        aoo = AOO_API(_n, _m, n0, m0, da, db, dc, pa, ngroup, read, c_init=0.1)
        #s, r = aoo.converge()
        #aoo.fraudar_v()
        #aoo.spoken()
