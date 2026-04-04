import numpy as np

class PricerClass:
    def __init__(self, alpha_r, alpha_m, alpha_l,
                 sigma_m, sigma_l, rho, mu):
        self.alpha_r = alpha_r
        self.alpha_m = alpha_m
        self.alpha_l = alpha_l
        self.sigma_m = sigma_m
        self.sigma_l = sigma_l
        self.rho = rho
        self.mu = mu

    def updParams(self,
              alpha_r=None, alpha_m=None, alpha_l=None,
              sigma_m=None, sigma_l=None,
              rho=None, mu=None):

        if alpha_r is not None:
            self.alpha_r = alpha_r
        if alpha_m is not None:
            self.alpha_m = alpha_m
        if alpha_l is not None:
            self.alpha_l = alpha_l
        if sigma_m is not None:
            self.sigma_m = sigma_m
        if sigma_l is not None:
            self.sigma_l = sigma_l
        if rho is not None:
            self.rho = rho
        if mu is not None:
            self.mu = mu

    def bVector(self, tau):
        b_1 = (1 - np.exp(-self.alpha_r * tau)) / (self.alpha_r * tau)
        b_2 = (1 - np.exp(-self.alpha_m * tau)) / (self.alpha_m * tau)
        b_3 = (1 - np.exp(-self.alpha_l * tau)) / (self.alpha_l * tau)
        return np.array([b_1, b_2, b_3])
    
    def aMatrix(self):

        a_11 = 1
        a_12 = 1
        a_13 = 1
        a_21 = 0
        a_22 = (self.alpha_r - self.alpha_m)/self.alpha_r
        a_23 = (self.alpha_r - self.alpha_l)/self.alpha_r
        a_31 = 0
        a_32 = 0
        a_33 = (self.alpha_r - self.alpha_l)*(self.alpha_m - self.alpha_l)/(self.alpha_r * self.alpha_m)
        return np.array([[a_11, a_12, a_13], [a_21, a_22, a_23], [a_31, a_32, a_33]])
    
    def factorLoadings(self, tau):
        b = self.bVector(tau)
        a = self.aMatrix()
        a_inv = np.linalg.inv(a)
        return b @ a_inv
    
    def omegaMatrix(self):
        omega_21 = self.rho * self.sigma_m
        omega_22 = np.sqrt(1 - self.rho**2) * self.sigma_m
        omega_31 = self.sigma_l
        return np.array([[0, 0, 0], [omega_21, omega_22, 0], [omega_31, 0, 0]])
        
    def convexityTerm(self, tau):
        alpha = np.array([self.alpha_r, self.alpha_m, self.alpha_l])
        a = self.aMatrix()
        a_inv = np.linalg.inv(a)
        
        omega = self.omegaMatrix()
        sigmaij_matrix = a_inv @ omega @ omega.T @ a_inv.T
        
        outputSum = 0
        for i in range(3):
            for j in range(3):
                mult_factor = sigmaij_matrix[i,j] / (2 * alpha[i] * alpha[j])  
                loadingSum = 1 - \
                    (1 - np.exp(-alpha[i] * tau)) / (alpha[i] * tau) - \
                    (1 - np.exp(-alpha[j] * tau)) / (alpha[j] * tau) + \
                    (1 - np.exp(- (alpha[i] + alpha[j]) * tau)) / ((alpha[i] + alpha[j]) * tau)
                
                outputSum += mult_factor * loadingSum
        return outputSum
    
    def convexityLimit(self):
        alpha = np.array([self.alpha_r, self.alpha_m, self.alpha_l])
        a = self.aMatrix()
        a_inv = np.linalg.inv(a)
        
        omega = self.omegaMatrix()
        sigmaij_matrix = a_inv @ omega @ omega.T @ a_inv.T
        outputSum = 0
        for i in range(3):
            for j in range(3):
                mult_factor = sigmaij_matrix[i,j] / (2 * alpha[i] * alpha[j])  
                loadingSum = 1
                outputSum += mult_factor * loadingSum
        return outputSum
    
    def factorLoadings_forwards(self, tau, deltaTau):
        
        bVector_tauPrime = self.bVector(tau = tau + deltaTau)
        bVector_tau = self.bVector(tau = tau)
        a = self.aMatrix()
        a_inv = np.linalg.inv(a)
        return (bVector_tauPrime - bVector_tau) @ a_inv * 1/deltaTau
    
    def convexityTerm_forwards(self, tau, deltaTau):
        cTerm_tau = self.convexityTerm(tau = tau)
        cTerm_tauPrime = self.convexityTerm(tau = tau + deltaTau)
        return (cTerm_tauPrime - cTerm_tau)/deltaTau


    def bondYield(self, tau, r, m, l):
        factor_loadings = self.factorLoadings(tau)
        convexity_term = self.convexityTerm(tau)

        return self.mu * (1 - np.ones(3) @ factor_loadings) + factor_loadings @ np.array([r, m, l]) - convexity_term
    
    def termStructure(self, maturities, factors):
        term_structure = []
        for maturity in maturities:
            term_structure.append(self.bondYield(maturity, factors[0], factors[1], factors[2]))
        return np.array(term_structure)
    
    def forwardRate(self, tau, deltaTau, r, m, l):
        frontYield = self.bondYield(tau = tau, r = r, m = m, l = l)
        backYield = self.bondYield(tau = tau+deltaTau, r = r, m = m, l = l)
        forwardYield = ((tau + deltaTau) * backYield - (tau) * frontYield)/ deltaTau
        return forwardYield
    
    def forwardTermStructure(self, deltaTau, maturities, factors):
        term_structure = []
        for maturity in maturities:
            term_structure.append(self.forwardRate(tau = maturity, deltaTau = deltaTau,
                                                   r = factors[0], m = factors[1], l = factors[2]))
        return np.array(term_structure)
    
    def amountOfRisk(self, tau, deltaTau, n_steps = 1000):

        omega = self.omegaMatrix()
        grid = np.linspace(0.0, deltaTau)
        vals = np.zeros_like(grid)
        for k, s in enumerate(grid):
            z = tau + deltaTau - s
            gamma = self.factorLoadings(z)
            gamma3 = gamma[2]
            vals[k] = (z * gamma3 * self.sigma_l - 0.5* z**2 * (gamma @ omega @ omega.T @ gamma.T))

        rp = np.trapz(vals, grid)
        return rp
    
    
    
