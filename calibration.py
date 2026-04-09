import numpy as np
from pricing.pricer import PricerClass
from scipy.optimize import minimize

class Calibration():
    def __init__(self, termStructurePath, forwardTermStructurePath, allForwardsPath, 
                 useInputForwards, maturities, fairly_priced, fairly_priced_fwd, 
                 fwd_deltaTau, alpha_r = 1.0547, lossDecayFactor = 0.8):
        self.termStructurePath = np.asarray(termStructurePath)
        self.forwardTermStructurePath = np.asarray(forwardTermStructurePath)
        self.allForwardsPath = allForwardsPath
        self.useInputForwards = useInputForwards
        self.maturities = maturities
        self.fairly_priced = fairly_priced
        self.fairly_priced_fwd = fairly_priced_fwd
        self.fwd_deltaTau = fwd_deltaTau
        self.fairly_priced_fwdKeys = [(tau, fwd_deltaTau) for tau in fairly_priced_fwd]
        if len(self.fairly_priced_fwd) != 2:
            raise ValueError("fairly_priced_fwd must contain exactly two forward starts")
        
        self.alpha_r = alpha_r
        self.pricer = PricerClass(alpha_r = alpha_r, alpha_m = None, alpha_l = None, sigma_m = None, sigma_l = None, rho = None, mu = None)
        self.lossDecayFactor = lossDecayFactor
        self.lossDecayFunction = lambda x: lossDecayFactor**(x/252)
        
    # General Utils

    def shortRateLoading(self, tau, alpha_r = None):
        '''
        returns the loading of the short rate for subtracting the short rate * its loading purposes
        '''
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        return (1 - np.exp(-alpha_r * tau)) / (alpha_r * tau)
    
    def subtractShortRate(self, alpha_r = None):
        '''
        subtracts the short rate * its loading from each spot rate series
        '''
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r

        ts = self.termStructurePath.copy()
        loadings = [self.shortRateLoading(tau = x, alpha_r = alpha_r) for x in self.maturities]
        shortrate = ts[:, 0]
        subtractedYields = ts[:, 1:]- np.outer(shortrate, np.array(loadings))
        return subtractedYields
    
    def regression(self, alpha_r = None):
        '''
        regresses each spot rate onto the two benchmarks (2y and 10y) and returns the regression betas for each spot rate
        '''
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
        '''
        this goes into the objective function, it computeds the model implied reversions of a given tau 
        point for a given alpha_m and alpha_l
        '''
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        
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
        '''
        computes the model implied variance of the change in the yield of a given tau point for given parameters
        '''
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho)
        omega = pricer.omegaMatrix()
        omegaTilde = omega[1:3, 0:2]
        gammaTildeTau = pricer.factorLoadings(tau = tau)[1:3]
        return gammaTildeTau @ omegaTilde @ omegaTilde.T @ gammaTildeTau.T
    
    def modelImpliedVarcov(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho):
        '''
        computes the full model implied varcov matrix of changes in yields
        '''
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
        '''
        computes the empirical variance of changes in a given tau point, net of the contribution of the short rate
        '''
        subtractedYields = self.subtractShortRate(alpha_r = alpha_r)
        targetYield = subtractedYields[:, list(self.maturities).index(tau)]
        targetYield_chg = np.diff(targetYield)
        return np.var(targetYield_chg, ddof = 1)
    
    def empiricalVarcov(self, alpha_r):
        '''
        computes the full varcov matrix of changes in all tau points, net of the contribution of the short rate
        '''
        targetYields = self.subtractShortRate(alpha_r = alpha_r)
        targetYields_chg = np.diff(targetYields, axis = 0)
        return np.cov(targetYields_chg, rowvar = False, ddof = 1)
    
    def objectiveFunction_sigma(self, x, alpha_r, alpha_m, alpha_l):
        '''
        matching model-implied VARIANCES to empirical VARIANCES
        '''
        sigma_m, sigma_l, rho = x
        total = 0
        for tau in self.maturities:
            modelVar = self.modelImpliedVariance(tau, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho)
            empiricalVar = self.empiricalVariance(tau, alpha_r)
            total += (modelVar - empiricalVar)**2
        return total
    
    def objectiveFunction_sigma_cov(self, x, alpha_r, alpha_m, alpha_l):
        '''
        matching model-implied FULL VARCOV to empirical VARCOV
        '''
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
        '''
        build the varcov given three inputs of a cholesky given from the optimizer
        '''    
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
        '''
        computes the model implied varcov given a candidate x from the optimizer, fed into choleskyCovariance
        to build the model-implied varcov
        '''
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l)
        
        gammaTilde = np.array([
            pricer.factorLoadings(tau=tau)[1:3]
            for tau in self.maturities
        ]) 

        Sigma_x = self.choleskyCovariance(x)                       # 2x2

        return gammaTilde @ Sigma_x @ gammaTilde.T   # NxN

    def objectiveFunction_sigma_chol(self, x, alpha_r, alpha_m, alpha_l):
        '''
        matching model-implied FULL VARCOV to empirical VARCOV, but search space is the cholesky space
        to ensure positive definitness without constrains
        '''
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
        '''
        get latent factors for given alpha, sigma and mu
        '''
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
    
    def fittedYieldsFromMu(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, extraction):
        '''
        get the fitted yields given a candidate mu, once you calibrated alpha and sigma
        extraction == 'spot' uses fairly priced spot rates (e.g. 2y, 10y) to extract m_t, l_t
        extraction == 'fwd' uses fairly priced forwards (e.g. 2y1y, 10y1y) to extract m_t, l_t
        '''
        mu = float(mu)
        if extraction == 'spot':
            lf = self.extractLatentFactors(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu)
        elif extraction == 'fwd':
            lf = self.extractLatentFactors_fwd(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, deltaTau= self.fwd_deltaTau)
        else:
            raise ValueError("extraction must be 'spot' or 'fwd'")
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho, mu = mu)
        
        shortRatePath = self.termStructurePath[:, 0]
        fitted = []
        for t in range(len(shortRatePath)):
            curve = pricer.termStructure(maturities=self.maturities, factors= np.array([shortRatePath[t], lf[t, 0], lf[t, 1]]))
            fitted.append(curve)
        return np.array(fitted)
    
    def objectiveFunction_mu(self, x, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, extraction, lossDecayFactor = None):
        if lossDecayFactor:
            lossDecayFunction = lambda x: lossDecayFactor**(x/252)
        else:
            lossDecayFunction = self.lossDecayFunction

        mu = float(x[0]) if np.ndim(x) > 0 else float(x)
        fittedYields = self.fittedYieldsFromMu(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, extraction)
        trueYields = self.termStructurePath[:, 1:]
        decayWeights = [lossDecayFunction(x) for x in range(fittedYields.shape[0] - 1, -1, -1)]
        return np.dot(decayWeights, np.sum((fittedYields - trueYields)**2, axis = 1))
    
    def calibrateMu(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, initialGuess = 0.0, extraction = 'spot', lossDecayFactor = None):
        if not lossDecayFactor:
            lossDecayFactor = self.lossDecayFactor
        result = minimize(
            self.objectiveFunction_mu,
            x0=initialGuess,
            args=(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, extraction, lossDecayFactor),
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

    def extractLatentFactors_fwd(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, deltaTau = 1):
        
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho, mu = mu)
        
        if self.useInputForwards:
            midForwardSeries = self.allForwardsPath[self.fairly_priced_fwdKeys[0]]
            longForwardSeries = self.allForwardsPath[self.fairly_priced_fwdKeys[1]]
            #midForwardSeries = self.forwardTermStructurePath[:, np.where(self.maturities == self.fairly_priced_fwd[0])[0][0]]
            #longForwardSeries = self.forwardTermStructurePath[:, np.where(self.maturities == self.fairly_priced_fwd[1])[0][0]]
        else:
            midForwardSeries = self.observedForwardRateSeries(tau = self.fairly_priced_fwd[0], deltaTau= deltaTau)
            longForwardSeries = self.observedForwardRateSeries(tau = self.fairly_priced_fwd[1], deltaTau= deltaTau)
        
        ts = self.termStructurePath
        shortRate = ts[:, 0]
        midForwardSeries_sub = midForwardSeries - pricer.factorLoadings_forwards(tau = self.fairly_priced_fwd[0], deltaTau= deltaTau)[0] * shortRate
        longForwardSeries_sub = longForwardSeries - pricer.factorLoadings_forwards(tau = self.fairly_priced_fwd[1], deltaTau= deltaTau)[0] * shortRate

        benchmarkForwards = np.column_stack((midForwardSeries_sub, longForwardSeries_sub))
        benchFwdLoadings = np.row_stack((pricer.factorLoadings_forwards(tau = self.fairly_priced_fwd[0], deltaTau= deltaTau)[1:3],
                                         pricer.factorLoadings_forwards(tau = self.fairly_priced_fwd[1], deltaTau= deltaTau)[1:3]))
        benchFwdLoadings_inv = np.linalg.inv(benchFwdLoadings)
        
        constantVector = np.array([1 - np.sum(pricer.factorLoadings_forwards(tau = x, deltaTau = deltaTau)) for x in self.fairly_priced_fwd])
        convexityTerms = np.array([pricer.convexityTerm_forwards(tau = x, deltaTau= deltaTau) for x in self.fairly_priced_fwd])

        latentFactors = (benchmarkForwards - mu * constantVector + convexityTerms) @ benchFwdLoadings_inv.T
        return latentFactors

    # functions to calibrate alpha and sigma via forwards

    def shortRateLoading_fwd(self, tau, deltaTau, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r

        b1_tau = (1 - np.exp(-alpha_r * tau)) / (alpha_r * tau)
        b1_tau_prime = (1 - np.exp(-alpha_r * (tau + deltaTau))) / (alpha_r * (tau + deltaTau))

        return ((tau + deltaTau) * b1_tau_prime - tau * b1_tau) / deltaTau

    def subtractShortRate_fwd(self, alpha_r = None):
        '''
        subtracts the short rate * its loading from each forward rate series
        '''
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r

        allForwardsPath = self.allForwardsPath.copy()
        shortrate = self.termStructurePath[:, 0]
        allForwardsPath_subtracted = {key: allForwardsPath[key] - self.shortRateLoading_fwd(
            tau = key[0], deltaTau = key[1], alpha_r = alpha_r
            ) * shortrate for key in allForwardsPath.keys()}
        return allForwardsPath_subtracted        

    def alphaComparison_fwd(self, tau, deltaTau, alpha_m, alpha_l, alpha_r = None):
        '''
        model implied regression betas of all forwards onto the two fairly priced forwards
        '''

        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        pricer = self.pricer
        pricer.updParams(alpha_m = alpha_m, alpha_l = alpha_l, alpha_r = alpha_r)
        
        benchForwardLoadings = np.array([pricer.factorLoadings_forwards(tau=x, deltaTau=y)[1:3] for x, y in self.fairly_priced_fwdKeys])
        gammaTildeTau = pricer.factorLoadings_forwards(tau = tau, deltaTau= deltaTau)[1:3]
        
        return gammaTildeTau @ np.linalg.inv(benchForwardLoadings)
    
    def regression_fwd(self, alpha_r = None):
        '''
        regresses each forward rate onto the two benchmarks
        '''
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        allForwardsPath_sub = self.subtractShortRate_fwd(alpha_r = alpha_r)
        allForwardsPath_sub = {key: np.diff(allForwardsPath_sub[key]) for key in allForwardsPath_sub.keys()}
        fairlyPriced = self.fairly_priced_fwdKeys
        betaStore = {}
        X = np.column_stack([allForwardsPath_sub[fpKey] for fpKey in fairlyPriced])
        for key in allForwardsPath_sub.keys():            
            y = allForwardsPath_sub[key]
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            betaStore[key] = beta
        return betaStore

    def objectiveFunction_firstStep_fwd(self, x, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        u, v = x
        alpha_m = u * alpha_r
        alpha_l = v * alpha_m
        total = 0
        regressionBetas = self.regression_fwd(alpha_r=alpha_r)
        for key in regressionBetas.keys():
            betaTau = regressionBetas[key]
            alphaComparison = self.alphaComparison_fwd(tau = key[0], deltaTau= key[1], alpha_m = alpha_m, alpha_l = alpha_l, alpha_r = alpha_r)
            total += np.linalg.norm(betaTau - alphaComparison)**2
        return total
    
    def calibrateAlphaFirstStep_fwd(self, initialGuess, alpha_r = None):
        alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        bounds=[(1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6)]

        result = minimize(self.objectiveFunction_firstStep_fwd, 
                          x0=initialGuess, 
                          args=(alpha_r,),
                          bounds=bounds,
                          method = 'L-BFGS-B')
        alpha_m = result.x[0] * alpha_r
        alpha_l = result.x[1] * alpha_m
        return {'alpha_m': alpha_m, 'alpha_l': alpha_l, 'loss': result.fun, 'success': result.success, 'message': result.message}
    
    def calibrateAlphaSecondStep_fwd(self, grid, initialGuess = [0.5, 0.5]):
        best = None
        for alpha_r in grid:
            
            calibratedAlphas = self.calibrateAlphaFirstStep_fwd(initialGuess = initialGuess,
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
    
    # Functions to calibrate sigmas via forwards

    def modelImpliedVariance_fwd(self, tau, deltaTau, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho):
        '''
        computes the model implied variance of the change in a given forward rate
        '''
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho)
        omega = pricer.omegaMatrix()
        omegaTilde = omega[1:3, 0:2]
        gammaTildeTauStar = pricer.factorLoadings_forwards(tau = tau, deltaTau= deltaTau)[1:3]
        return gammaTildeTauStar @ omegaTilde @ omegaTilde.T @ gammaTildeTauStar.T
    
    def modelImpliedVarcov_fwd(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho):
        '''
        computes the full model implied varcov matrix of changes in forward rates
        '''
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho)
        omega = pricer.omegaMatrix()
        omegaTilde = omega[1:3, 0:2]   # 2x2

        gammaTilde = np.array([
            pricer.factorLoadings_forwards(tau=key[0], deltaTau=key[1])[1:3]
            for key in self.allForwardsPath.keys()
        ]) 

        return gammaTilde @ omegaTilde @ omegaTilde.T @ gammaTilde.T   # NxN

    def empiricalVariance_fwd(self, tau, deltaTau, alpha_r):
        '''
        computes the empirical variance of changes in a given forward rate, net of the contribution of the short rate
        '''
        allForwardsPath_sub = self.subtractShortRate_fwd(alpha_r = alpha_r)
        targetForward = allForwardsPath_sub[(tau, deltaTau)]
        targetForward_chg = np.diff(targetForward)
        return np.var(targetForward_chg, ddof = 1)

    def empiricalVarcov_fwd(self, alpha_r):
        '''
        computes the full varcov matrix of changes in forward rates, net of the contribution of the short rate
        '''
        allForwardsPath_sub = self.subtractShortRate_fwd(alpha_r = alpha_r)
        targetForwards = np.column_stack([allForwardsPath_sub[key] for key in self.allForwardsPath.keys()])
        targetForwards_chg = np.diff(targetForwards, axis = 0)
        return np.cov(targetForwards_chg, rowvar = False, ddof = 1)
    
    def objectiveFunction_sigma_fwd(self, x, alpha_r, alpha_m, alpha_l):
        '''
        matching model-implied VARIANCES to empirical VARIANCES for forwards
        '''
        sigma_m, sigma_l, rho = x
        total = 0
        for key in self.allForwardsPath.keys():
            modelVar = self.modelImpliedVariance_fwd(tau=key[0], deltaTau=key[1], alpha_r=alpha_r, alpha_m=alpha_m, alpha_l=alpha_l, sigma_m=sigma_m, sigma_l=sigma_l, rho=rho)
            empiricalVar = self.empiricalVariance_fwd(tau=key[0], deltaTau=key[1], alpha_r=alpha_r)
            total += (modelVar - empiricalVar)**2
        return total

    def calibrateSigma_fwd(self, alpha_r, alpha_m, alpha_l, initialGuess = [0.1, 0.1, 0]):
        bounds = [(1e-6, None), (1e-6, None), (-0.999, 0.999)]
        result = minimize(self.objectiveFunction_sigma_fwd, 
                          x0=initialGuess, 
                          args=(alpha_r, alpha_m, alpha_l),
                          bounds=bounds,
                          method = 'L-BFGS-B')
        sigma_m, sigma_l, rho = result.x
        print(f'sigma_m: {sigma_m}, sigma_l: {sigma_l}, rho: {rho}, loss: {result.fun}, success: {result.success}, message: {result.message}')
        return {'sigma_m': sigma_m, 'sigma_l': sigma_l, 'rho': rho, 'loss': result.fun, 'success': result.success, 'message': result.message}

    def objectiveFunction_sigma_cov_fwd(self, x, alpha_r, alpha_m, alpha_l):
        '''
        matching model-implied FULL VARCOV to empirical VARCOV for forwards
        '''
        sigma_m, sigma_l, rho = x

        modelCov = self.modelImpliedVarcov_fwd(
            alpha_r=alpha_r,
            alpha_m=alpha_m,
            alpha_l=alpha_l,
            sigma_m=sigma_m,
            sigma_l=sigma_l,
            rho=rho
        )

        empiricalCov = self.empiricalVarcov_fwd(alpha_r=alpha_r)

        diff = modelCov - empiricalCov
        return np.sum(diff**2)
    
    def calibrateSigmaCov_fwd(self, alpha_r, alpha_m, alpha_l, initialGuess=[0.1, 0.1, 0.0]):
        bounds = [(1e-6, None), (1e-6, None), (-0.999, 0.999)]

        result = minimize(
            self.objectiveFunction_sigma_cov_fwd,
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

        return {'sigma_m': sigma_m, 'sigma_l': sigma_l, 'rho': rho, 'loss': result.fun, 'success': result.success, 'message': result.message}

    def modelImpliedVarcov_chol_fwd(self, alpha_r, alpha_m, alpha_l, x):
        '''
        computes the model implied varcov given a candidate x from the optimizer, fed into choleskyCovariance
        to build the model-implied varcov for forwards
        '''
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l)
        
        gammaTilde = np.array([
            pricer.factorLoadings_forwards(tau=key[0], deltaTau=key[1])[1:3]
            for key in self.allForwardsPath.keys()
        ]) 

        Sigma_x = self.choleskyCovariance(x)                       # 2x2

        return gammaTilde @ Sigma_x @ gammaTilde.T   # NxN

    def objectiveFunction_sigma_chol_fwd(self, x, alpha_r, alpha_m, alpha_l):
        '''
        matching model-implied FULL VARCOV to empirical VARCOV for forwards, but search space is the cholesky space
        to ensure positive definitness without constrains
        '''
        modelCov = self.modelImpliedVarcov_chol_fwd(alpha_r, alpha_m, alpha_l, x)
        empiricalCov = self.empiricalVarcov_fwd(alpha_r)

        diff = modelCov - empiricalCov
        return np.sum(diff**2)
    
    def calibrateSigmaChol_fwd(self, alpha_r, alpha_m, alpha_l, initialGuess=[-4.5, -4.5, 0.0]):
        result = minimize(
            self.objectiveFunction_sigma_chol_fwd,
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
    
    # calibrate mu to match the forward surface instead of spot

    def fittedForwardsFromMu(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, extraction):
        '''
        get the fitted forward rates given a candidate mu, once you calibrated alpha and sigma
        extraction == 'spot' uses fairly priced spot rates (e.g. 2y, 10y) to extract m_t, l_t
        extraction == 'fwd' uses fairly priced forwards (e.g. 2y1y, 10y1y) to extract m_t, l_t
        '''
        mu = float(mu)
        if extraction == 'spot':
            lf = self.extractLatentFactors(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu)
        elif extraction == 'fwd':
            lf = self.extractLatentFactors_fwd(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, deltaTau= self.fwd_deltaTau)
        else:
            raise ValueError("extraction must be 'spot' or 'fwd'")
        pricer = self.pricer
        pricer.updParams(alpha_r = alpha_r, alpha_m = alpha_m, alpha_l = alpha_l, sigma_m = sigma_m, sigma_l = sigma_l, rho = rho, mu = mu)
        shortRatePath = self.termStructurePath[:, 0]

        fittedYields = np.array([
            pricer.termStructure(
                maturities=self.maturities,
                factors=np.array([shortRatePath[t], lf[t, 0], lf[t, 1]])
            )
            for t in range(len(shortRatePath))
        ])
        maturity_to_idx = {tau: i for i, tau in enumerate(self.maturities)}
        fittedForwards = np.column_stack([
            (
                (tau + deltaTau) * fittedYields[:, maturity_to_idx[tau + deltaTau]]
                - tau * fittedYields[:, maturity_to_idx[tau]]
            ) / deltaTau
            for tau, deltaTau in self.allForwardsPath.keys()
        ])
        return fittedForwards
            
    
    def objectiveFunction_mu_fwd(self, x, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, extraction, lossDecayFactor = None):
        if lossDecayFactor:
            lossDecayFunction = lambda x: lossDecayFactor**(x/252)
        else:
            lossDecayFunction = self.lossDecayFunction

        mu = float(x[0]) if np.ndim(x) > 0 else float(x)
        fittedForwards = self.fittedForwardsFromMu(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, extraction)
        trueForwards = np.column_stack([self.allForwardsPath[key] for key in self.allForwardsPath.keys()])
        #trueForwards = {key: self.allForwardsPath[key] for key in self.allForwardsPath.keys()}
        decayWeights = np.array([lossDecayFunction(i) for i in range(fittedForwards.shape[0] - 1, -1, -1)])
        return np.dot(decayWeights, np.sum((fittedForwards - trueForwards)**2, axis=1))
    
    def calibrateMu_fwd(self, alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, initialGuess = 0.0, extraction = 'fwd', lossDecayFactor = None):
        if not lossDecayFactor:
            lossDecayFactor = self.lossDecayFactor
        result = minimize(
            self.objectiveFunction_mu_fwd,
            x0=initialGuess,
            args=(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, extraction, lossDecayFactor),
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
    
    def lambdaFromForwards_corrected(self, tau, tauPrime, deltaTau, curve,
                       alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu,
                       n_steps=1000):
        '''
        corrects the mistake of taking out lambda as if it multiplied the convexity term in Eq A9.27 of the appendix
        '''
        pricer = self.pricer
        pricer.updParams(alpha_r=alpha_r, alpha_m=alpha_m, alpha_l=alpha_l,
                        sigma_m=sigma_m, sigma_l=sigma_l, rho=rho, mu=mu)
        
        forwardTau = self.observedForwardRate(tau=tau, deltaTau=deltaTau, curve = curve)
        forwardTauPrime = self.observedForwardRate(tau=tauPrime, deltaTau=deltaTau, curve = curve)

        rpTau_drift = pricer.amountOfRisk_drift(tau=tau, deltaTau=deltaTau, n_steps=n_steps)
        rpTauPrime_drift = pricer.amountOfRisk_drift(tau=tauPrime, deltaTau=deltaTau, n_steps=n_steps)

        rpTau_conv =  pricer.amountOfRisk_convexity(tau=tau, deltaTau=deltaTau, n_steps=n_steps)
        rpTauPrime_conv = pricer.amountOfRisk_convexity(tau=tauPrime, deltaTau=deltaTau, n_steps=n_steps)

        return ((forwardTauPrime - forwardTau)*deltaTau + (rpTauPrime_conv - rpTau_conv))/(rpTauPrime_drift - rpTau_drift)
    
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
    
    def lambdaRegression_twoPremia(self, tau_0, tauList, deltaTau, curve,
                                   alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, n_steps=1000):
        '''
        cross-sectional regression to determine risk premia to two factors
        '''
        tauList = [x for x in tauList if x != tau_0]
        pricer = self.pricer
        pricer.updParams(alpha_r=alpha_r, alpha_m=alpha_m, alpha_l=alpha_l,
                        sigma_m=sigma_m, sigma_l=sigma_l, rho=rho, mu=mu)
        
        forwardTau_0 = self.observedForwardRate(tau=tau_0, deltaTau=deltaTau, curve = curve)
        driftMedium_tau0 = pricer.amountOfRisk_drift_medium(tau = tau_0, deltaTau=deltaTau, n_steps = n_steps)
        driftLong_tau0 = pricer.amountOfRisk_drift_long(tau = tau_0, deltaTau=deltaTau, n_steps = n_steps)
        convexity_tau0 = pricer.amountOfRisk_convexity(tau = tau_0, deltaTau=deltaTau, n_steps = n_steps)
        F = np.array([
            self.observedForwardRate(tau = x, deltaTau=deltaTau, curve = curve) - forwardTau_0 + 
            (pricer.amountOfRisk_convexity(tau = x, deltaTau=deltaTau, n_steps = n_steps) - convexity_tau0) * 1/deltaTau
        for x in tauList])

        D = np.array([
            [pricer.amountOfRisk_drift_medium(tau = x, deltaTau= deltaTau, n_steps= n_steps) - driftMedium_tau0, 
             pricer.amountOfRisk_drift_long(tau = x, deltaTau= deltaTau, n_steps = n_steps) - driftLong_tau0]
        for x in tauList]) * 1/deltaTau

        lambdaOLS = np.linalg.inv(D.T @ D) @ D.T @ F
        return (F, D, lambdaOLS)

    def lambdaRegression_twoPremia_ts(self, tauList, deltaTau,
                                   alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, n_steps=1000):
        '''
        regresses realized returns of the strategy exposed to risk premia onto its loadings. in practice:
        R_t^tau = f_t(tau) - r_(t+1)
        Regresses R_t^tau + convexity term = loading_m * lambda_m + loading_l * lambda_l
        '''
        
        pricer = self.pricer
        pricer.updParams(alpha_r=alpha_r, alpha_m=alpha_m, alpha_l=alpha_l,
                        sigma_m=sigma_m, sigma_l=sigma_l, rho=rho, mu=mu)
        
        shortRateSeries = self.termStructurePath[:, 0]
        y = []
        X = []
        for tau in tauList:
            forwardSeries_tau = self.observedForwardRateSeries(tau = tau, deltaTau = deltaTau)
            realizedReturn = (forwardSeries_tau[:-1] - shortRateSeries[1:]) * deltaTau
            driftMedium = pricer.amountOfRisk_drift_medium(tau = tau, deltaTau=deltaTau, n_steps=n_steps)
            driftLong = pricer.amountOfRisk_drift_long(tau=tau, deltaTau=deltaTau, n_steps=n_steps)
            convexity = pricer.amountOfRisk_convexity(tau = tau, deltaTau=deltaTau, n_steps=n_steps)

            y.append(realizedReturn + convexity)
            X_tau = np.column_stack([
                np.full(realizedReturn.shape, driftMedium),
                np.full(realizedReturn.shape, driftLong)
            ])
            X.append(X_tau)
        
        y = np.concatenate(y)
        X = np.vstack(X)
        lambdaOLS = np.linalg.inv(X.T @ X) @ X.T @ y
        return y, X, lambdaOLS
    
    def lambdaRegression_twoPremia_ts_exp(self, tauList, deltaTau,
                                   alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu, 
                                   n_steps=1000, min_obs = 60):
        
        '''
        runs the same regressions as lambdaRegression_twoPremia_ts() but in expanding windows
        '''
        pricer = self.pricer
        pricer.updParams(alpha_r=alpha_r, alpha_m=alpha_m, alpha_l=alpha_l,
                        sigma_m=sigma_m, sigma_l=sigma_l, rho=rho, mu=mu)
        
        shortRateSeries = self.termStructurePath[:, 0]
        T = len(shortRateSeries) - 1
        y = []
        X = []
        for tau in tauList:
            forwardSeries_tau = self.observedForwardRateSeries(tau = tau, deltaTau = deltaTau)
            realizedReturn = (forwardSeries_tau[:-1] - shortRateSeries[1:]) * deltaTau
            driftMedium = pricer.amountOfRisk_drift_medium(tau = tau, deltaTau=deltaTau, n_steps=n_steps)
            driftLong = pricer.amountOfRisk_drift_long(tau=tau, deltaTau=deltaTau, n_steps=n_steps)
            convexity = pricer.amountOfRisk_convexity(tau = tau, deltaTau=deltaTau, n_steps=n_steps)

            y.append(realizedReturn + convexity)
            X_tau = np.column_stack([
                np.full(realizedReturn.shape, driftMedium),
                np.full(realizedReturn.shape, driftLong)
            ])
            X.append(X_tau)
        
        y_mat = np.column_stack(y)
        X1_mat = np.column_stack([X[:, 0] for X in X])
        X2_mat = np.column_stack([X[:, 1] for X in X])

        lambda_series = np.full((T, 2), np.nan)
        r2_series = np.full(T, np.nan)

        for t in range(min_obs - 1, T):
            # expanding window 0..t
            y_win = y_mat[:t+1, :].reshape(-1)
            X_win = np.column_stack([
                X1_mat[:t+1, :].reshape(-1),
                X2_mat[:t+1, :].reshape(-1)
            ])

            beta, _, _, _ = np.linalg.lstsq(X_win, y_win, rcond=None)
            lambda_series[t, :] = beta

            fitted = X_win @ beta
            rss = np.sum((y_win - fitted) ** 2)
            tss = np.sum((y_win - y_win.mean()) ** 2)
            r2_series[t] = 1 - rss / tss if tss > 0 else np.nan

        return {
            "lambda_m_series": lambda_series[:, 0],
            "lambda_l_series": lambda_series[:, 1],
            "lambda_series": lambda_series,
            "r2_series": r2_series,
            "y_mat": y_mat,
            "X1_mat": X1_mat,
            "X2_mat": X2_mat,
            "tauList": tauList
        }
