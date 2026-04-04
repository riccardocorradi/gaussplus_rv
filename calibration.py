import numpy as np
from pricing.pricer import PricerClass
from scipy.optimize import minimize

class Calibration():
    def __init__(self, termStructurePath, maturities, fairly_priced, alpha_r = 1.0547, lossDecayFactor = 0.8):
        self.termStructurePath = np.asarray(termStructurePath)
        self.maturities = maturities
        self.fairly_priced = fairly_priced
        self.alpha_r = alpha_r
        self.pricer = PricerClass(alpha_r = alpha_r, alpha_m = None, alpha_l = None, sigma_m = None, sigma_l = None, rho = None, mu = None)
        self.lossDecayFactor = lossDecayFactor
        self.lossDecayFunction = lambda x: lossDecayFactor**(x/252)
    
    # General Utils

    def shortRateLoading(self, tau, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        return (1 - np.exp(-alpha_r * tau)) / (alpha_r * tau)
    
    def subtractShortRate(self, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r

        ts = self.termStructurePath.copy()
        loadings = [self.shortRateLoading(tau = x, alpha_r = alpha_r) for x in self.maturities]
        shortrate = ts[:, 0]
        subtractedYields = ts[:, 1:]- np.outer(shortrate, np.array(loadings))
        return subtractedYields
    
    def regression(self, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        subtractedYields = self.subtractShortRate(alpha_r=alpha_r)
        subtractedYields = np.diff(subtractedYields, axis = 0)
        fairlyPricedIdx = [list(self.maturities).index(x) for x in self.fairly_priced]
        betaStore = []
        for tau in range(subtractedYields.shape[1]):            
            X = subtractedYields[:, fairlyPricedIdx]
            y = subtractedYields[:, tau]
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            betaStore.append(beta)
        return betaStore
    
    def bVector(self, tau, alpha_m, alpha_l, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        b_1 = (1 - np.exp(-alpha_r * tau)) / (alpha_r * tau)
        b_2 = (1 - np.exp(-alpha_m * tau)) / (alpha_m * tau)
        b_3 = (1 - np.exp(-alpha_l * tau)) / (alpha_l * tau)
        return np.array([b_1, b_2, b_3])
    
    def aMatrix(self, alpha_m, alpha_l, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        a_11 = 1
        a_12 = 1
        a_13 = 1
        a_21 = 0
        a_22 = (alpha_r - alpha_m)/alpha_r
        a_23 = (alpha_r - alpha_l)/alpha_r
        a_31 = 0
        a_32 = 0
        a_33 = (alpha_r - alpha_l)*(alpha_m - alpha_l)/(alpha_r * alpha_m)
        return np.array([[a_11, a_12, a_13], [a_21, a_22, a_23], [a_31, a_32, a_33]])
    
    def factorLoadings(self, tau, alpha_m, alpha_l, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        b = self.bVector(tau, alpha_m, alpha_l, alpha_r)
        a = self.aMatrix(alpha_m, alpha_l, alpha_r)
        a_inv = np.linalg.inv(a)
        return b @ a_inv

    # Functions to calibrate Alphas    

    def alphaComparison(self, tau, alpha_m, alpha_l, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        # this goes into the objective function, it computeds the model implied reversions of a given tau point for a given alpha_m and alpha_l
        pricer = self.pricer
        pricer.updParams(alpha_m = alpha_m, alpha_l = alpha_l, alpha_r = alpha_r)
        
        benchYieldLoadings = np.array([pricer.factorLoadings(tau = x)[1:3] for x in self.fairly_priced])
        gammaTildeTau = pricer.factorLoadings(tau = tau)[1:3]
        
        return gammaTildeTau @ np.linalg.inv(benchYieldLoadings)
    
    def objectiveFunction_firstStep(self, x, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        u, v = x
        alpha_m = u * alpha_r
        alpha_l = v * alpha_m
        total = 0
        regressionBetas = self.regression(alpha_r=alpha_r)
        for i in range(len(self.maturities)):
            betaTau = regressionBetas[i]
            alphaComparison = self.alphaComparison(tau = self.maturities[i], alpha_m = alpha_m, alpha_l = alpha_l, alpha_r = alpha_r)
            total += np.linalg.norm(betaTau - alphaComparison)**2
        return total
    
    def calibrateAlphaFirstStep(self, initialGuess, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        bounds=[(1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6)]

        result = minimize(self.objectiveFunction_firstStep, 
                          x0=initialGuess, 
                          args=(alpha_r,),
                          bounds=bounds,
                          method = 'L-BFGS-B')
        alpha_m = result.x[0] * alpha_r
        alpha_l = result.x[1] * alpha_m
        return {'alpha_m': alpha_m, 'alpha_l': alpha_l, 'loss': result.fun, 'success': result.success, 'message': result.message}
    
    def calibrateAlphaSecondStep(self, grid, initialGuess = [0.5, 0.5]):
        best = None
        for alpha_r in grid:
            
            calibratedAlphas = self.calibrateAlphaFirstStep(initialGuess = initialGuess,
                                                            alpha_r = alpha_r)
            if best is None or calibratedAlphas['loss'] < best['loss']:
                best = {
                    'alpha_r': alpha_r,
                    'alpha_m': calibratedAlphas['alpha_m'],
                    'alpha_l': calibratedAlphas['alpha_l'],
                    'loss': calibratedAlphas['loss'],
                    'success': calibratedAlphas['success'],
                    'message': calibratedAlphas['message']
                    }
                
                
        print(f'alpha_r: {best["alpha_r"]}, alpha_m: {best["alpha_m"]}, \
              alpha_l: {best["alpha_l"]}, loss: {best["loss"]},\
              success: {best["success"]}, message: {best["message"]}')
        return best
    
    # Functions to calibrate Sigmas

    def modelImpliedVariance(self, tau, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho):
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho)
        omega = pricer.omegaMatrix()
        omegaTilde = omega[1:3, 0:2]
        gammaTildeTau = pricer.factorLoadings(tau = tau)[1:3]
        return gammaTildeTau @ omegaTilde @ omegaTilde.T @ gammaTildeTau.T
    
    def modelImpliedVarcov(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho):
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho)
        omega = pricer.omegaMatrix()
        omegaTilde = omega[1:3, 0:2]   # 2x2

        gammaTilde = np.array([
            pricer.factorLoadings(tau=tau)[1:3]
            for tau in self.maturities
        ]) 

        return gammaTilde @ omegaTilde @ omegaTilde.T @ gammaTilde.T   # NxN
        
    def empiricalVariance(self, tau, alpha_r):
        subtractedYields = self.subtractShortRate(alpha_r = alpha_r)
        targetYield = subtractedYields[:, list(self.maturities).index(tau)]
        targetYield_chg = np.diff(targetYield)
        return np.var(targetYield_chg, ddof = 1)
    
    def empiricalVarcov(self, alpha_r):
        targetYields = self.subtractShortRate(alpha_r = alpha_r)
        targetYields_chg = np.diff(targetYields, axis = 0)
        return np.cov(targetYields_chg, rowvar = False, ddof = 1)
    
    def objectiveFunction_sigma(self, x, alpha_r, alpha_m, alpha_l):
        sigma_m, sigma_l, rho = x
        total = 0
        for tau in self.maturities:
            modelVar = self.modelImpliedVariance(tau, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho)
            empiricalVar = self.empiricalVariance(tau, alpha_r)
            total += (modelVar - empiricalVar)**2
        return total
    
    def objectiveFunction_sigma_cov(self, x, alpha_r, alpha_m, alpha_l):
        sigma_m, sigma_l, rho = x

        modelCov = self.modelImpliedVarcov(
            alpha_r=alpha_r,
            alpha_m=alpha_m,
            alpha_l=alpha_l,
            sigma_m=sigma_m,
            sigma_l=sigma_l,
            rho=rho
        )

        empiricalCov = self.empiricalVarcov(alpha_r=alpha_r)

        diff = modelCov - empiricalCov
        return np.sum(diff**2)
        
    def calibrateSigma(self, alpha_r, alpha_m, alpha_l, initialGuess = [0.1, 0.1, 0]):
        bounds = [(1e-6, None), (1e-6, None), (-0.999, 0.999)]
        result = minimize(self.objectiveFunction_sigma, 
                          x0=initialGuess, 
                          args=(alpha_r, alpha_m, alpha_l),
                          bounds=bounds,
                          method = 'L-BFGS-B')
        sigma_m, sigma_l, rho = result.x
        print(f'sigma_m: {sigma_m}, sigma_l: {sigma_l}, rho: {rho}, loss: {result.fun}, success: {result.success}, message: {result.message}')
        return {'sigma_m': sigma_m, 'sigma_l': sigma_l, 'rho': rho, 'loss': result.fun, 'success': result.success, 'message': result.message}

    def calibrateSigmaCov(self, alpha_r, alpha_m, alpha_l, initialGuess=[0.1, 0.1, 0.0]):
        bounds = [(1e-6, None), (1e-6, None), (-0.999, 0.999)]

        result = minimize(
            self.objectiveFunction_sigma_cov,
            x0=initialGuess,
            args=(alpha_r, alpha_m, alpha_l),
            bounds=bounds,
            method='L-BFGS-B'
        )

        sigma_m, sigma_l, rho = result.x

        print(
            f'sigma_m: {sigma_m}, sigma_l: {sigma_l}, rho: {rho}, '
            f'loss: {result.fun}, success: {result.success}, message: {result.message}'
        )

        return {
            'sigma_m': sigma_m,
            'sigma_l': sigma_l,
            'rho': rho,
            'loss': result.fun,
            'success': result.success,
            'message': result.message
        }

    def choleskyCovariance(self, x):    
        a, b, c = x
        l11 = np.exp(a)
        l22 = np.exp(b)
        l21 = c

        L = np.array([
            [l11, 0.0],
            [l21, l22]
        ])

        Sigma_x = L @ L.T
        return Sigma_x
    
    def modelImpliedVarcov_chol(self, alpha_r, alpha_m, alpha_l, x):
        
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l)
        
        gammaTilde = np.array([
            pricer.factorLoadings(tau=tau)[1:3]
            for tau in self.maturities
        ]) 

        Sigma_x = self.choleskyCovariance(x)                       # 2x2

        return gammaTilde @ Sigma_x @ gammaTilde.T   # NxN

    def objectiveFunction_sigma_chol(self, x, alpha_r, alpha_m, alpha_l):
        modelCov = self.modelImpliedVarcov_chol(alpha_r, alpha_m, alpha_l, x)
        empiricalCov = self.empiricalVarcov(alpha_r)

        diff = modelCov - empiricalCov
        return np.sum(diff**2)

    def calibrateSigmaChol(self, alpha_r, alpha_m, alpha_l, initialGuess=[-4.5, -4.5, 0.0]):
        result = minimize(
            self.objectiveFunction_sigma_chol,
            x0=initialGuess,
            args=(alpha_r, alpha_m, alpha_l),
            method='L-BFGS-B'
        )

        Sigma_x = self.choleskyCovariance(result.x)

        sigma_m = np.sqrt(Sigma_x[0, 0])
        sigma_l = np.sqrt(Sigma_x[1, 1])
        rho = Sigma_x[0, 1] / (sigma_m * sigma_l)

        print(
            f'sigma_m: {sigma_m}, sigma_l: {sigma_l}, rho: {rho}, '
            f'loss: {result.fun}, success: {result.success}, message: {result.message}'
        )

        return {
            'sigma_m': sigma_m,
            'sigma_l': sigma_l,
            'rho': rho,
            'Sigma_x': Sigma_x,
            'loss': result.fun,
            'success': result.success,
            'message': result.message
    }

    # Functions to calibrate Mu and extract factors by matching the 2y and 10y to m_t and l_t

    def extractLatentFactors(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu):
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho, mu = mu)

        subtractedYields = self.subtractShortRate(alpha_r = alpha_r)
        benchmarkYields = subtractedYields[:, [list(self.maturities).index(x) for x in self.fairly_priced]]
        benchYieldLoadings = np.array([pricer.factorLoadings(tau = x)[1:3] for x in self.fairly_priced])
        benchYieldLoadings_inv = np.linalg.inv(benchYieldLoadings)

        constantTerms = np.array([1 - np.sum(pricer.factorLoadings(tau = x)) for x in self.fairly_priced])
        convexityTerms = np.array([pricer.convexityTerm(tau = x) for x in self.fairly_priced])
        
        latentFactors = (benchmarkYields - mu * constantTerms + convexityTerms) @ benchYieldLoadings_inv.T
        
        return latentFactors
    
    def fittedYieldsFromMu(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu):
        mu = float(mu)
        lf = self.extractLatentFactors(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu)
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho, mu = mu)
        
        shortRatePath = self.termStructurePath[:, 0]
        fitted = []
        for t in range(len(shortRatePath)):
            curve = pricer.termStructure(maturities=self.maturities, factors= np.array([shortRatePath[t], lf[t, 0], lf[t, 1]]))
            fitted.append(curve)
        return np.array(fitted)
    
    def objectiveFunction_mu(self, x, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, lossDecayFactor = None):
        if lossDecayFactor:
            lossDecayFunction = lambda x: lossDecayFactor**(x/252)
        else:
            lossDecayFunction = self.lossDecayFunction

        mu = float(x[0]) if np.ndim(x) > 0 else float(x)
        fittedYields = self.fittedYieldsFromMu(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu)
        trueYields = self.termStructurePath[:, 1:]
        decayWeights = [lossDecayFunction(x) for x in range(fittedYields.shape[0] - 1, -1, -1)]
        return np.dot(decayWeights, np.sum((fittedYields - trueYields)**2, axis = 1))
    
    def calibrateMu(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, initialGuess = 0.0, lossDecayFactor = None):
        if not lossDecayFactor:
            lossDecayFactor = self.lossDecayFactor
        result = minimize(
            self.objectiveFunction_mu,
            x0=initialGuess,
            args=(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, lossDecayFactor),
            method='L-BFGS-B'
        )

        mu = result.x[0]

        print(
            f'mu: {mu}, loss: {result.fun}, success: {result.success}, message: {result.message}'
        )

        return {
            'mu': mu,
            'loss': result.fun,
            'success': result.success,
            'message': result.message
        }
    
    # Functions to calibrate Mu and extract factors by matching the 2y-forward 1y and 10y-forward 1y to m_t and l_t

    def marketForwardRateSeries(self, tau, deltaTau):
        assert (tau + deltaTau) in self.maturities, f'tau + deltaTau = {tau+deltaTau} must be in the maturity set'
        termStructurePath = self.termStructurePath[:, 1:]
        frontYield = termStructurePath[:, np.where(self.maturities == tau)[0][0]]
        backYield = termStructurePath[:, np.where(self.maturities == tau + deltaTau)[0][0]]
        return backYield        

    def extractLatentFactors_fwd(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu):
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho, mu = mu)


    # Risk premia calculation thru forwards

    def observedForwardRate(self, curve, tau, deltaTau):

        frontYield = np.interp(tau, self.maturities, curve)
        backYield = np.interp(tau + deltaTau, self.maturities, curve)
        forward = (backYield * (tau + deltaTau) - frontYield * tau) / (deltaTau)
        return forward
    
    def observedForwardRateSeries(self, tau, deltaTau):

        obsCurves = self.termStructurePath[:, 1:]
        output = []
        for curve in obsCurves:
            output.append(self.observedForwardRate(curve=curve, tau=tau, deltaTau=deltaTau))
        return np.array(output)

    def lambdaFromForwards(self, tau, tauPrime, deltaTau, curve,
                       alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu,
                       n_steps=1000):
        
        pricer = self.pricer
        pricer.updParams(alpha_r=alpha_r, alpha_m=alpha_m, alpha_l=alpha_l,
                        sigma_m=sigma_m, sigma_l=sigma_l, rho=rho, mu=mu)
        
        forwardTau = self.observedForwardRate(tau=tau, deltaTau=deltaTau, curve = curve)
        forwardTauPrime = self.observedForwardRate(tau=tauPrime, deltaTau=deltaTau, curve = curve)
        rpTau = pricer.amountOfRisk(tau=tau, deltaTau=deltaTau, n_steps=n_steps)
        rpTauPrime = pricer.amountOfRisk(tau=tauPrime, deltaTau=deltaTau, n_steps=n_steps)

        denom = rpTauPrime / deltaTau - rpTau / deltaTau
        if np.isclose(denom, 0.0):
            raise ValueError("Denominator too close to zero in lambdaFromForwards().")

        return (forwardTauPrime - forwardTau) / denom

    def lambdaSeriesFromForwards(self, tau, tauPrime, deltaTau,
                       alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu,
                       n_steps=1000):
        
        pricer = self.pricer
        pricer.updParams(alpha_r=alpha_r, alpha_m=alpha_m, alpha_l=alpha_l,
                        sigma_m=sigma_m, sigma_l=sigma_l, rho=rho, mu=mu)
        
        forwardTau = self.observedForwardRateSeries(tau=tau, deltaTau=deltaTau)
        forwardTauPrime = self.observedForwardRateSeries(tau=tauPrime, deltaTau=deltaTau)

        rpTau = pricer.amountOfRisk(tau=tau, deltaTau=deltaTau, n_steps=n_steps)
        rpTauPrime = pricer.amountOfRisk(tau=tauPrime, deltaTau=deltaTau, n_steps=n_steps)

        denom = rpTauPrime / deltaTau - rpTau / deltaTau
        if np.isclose(denom, 0.0):
            raise ValueError("Denominator too close to zero in lambdaFromForwards().")

        return (forwardTauPrime - forwardTau) / denom
    
    def expectedShortRateSeries(self, tau, tauPrime, deltaTau,
                            alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu,
                            n_steps=1000):
        """
        Time series of E_t^P[r_tau], using observed forwards at each time index t
        tauPrime and tau are the benchmark forwards, sufficiently far out in the long end
        delta Tau is the tenor
        """

        pricer = self.pricer
        pricer.updParams(alpha_r=alpha_r, alpha_m=alpha_m, alpha_l=alpha_l,
                        sigma_m=sigma_m, sigma_l=sigma_l, rho=rho, mu=mu)

        f_tau = self.observedForwardRateSeries(tau=tau, deltaTau=deltaTau)
        rp_tau = pricer.amountOfRisk(tau=tau, deltaTau=deltaTau, n_steps=n_steps)
        lambdaSeries = self.lambdaSeriesFromForwards(tau = tau, tauPrime= tauPrime, deltaTau= deltaTau, 
                                                     alpha_r=alpha_r, alpha_m=alpha_m, alpha_l=alpha_l,
                                                     sigma_m=sigma_m, sigma_l=sigma_l, rho=rho, mu=mu)

        return f_tau - lambdaSeries * (rp_tau / deltaTau)