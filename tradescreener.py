import pandas as pd
from itertools import combinations

class tradeScreener:

    def __init__(self, modelData, actualData, maturitySet):
        self.modelData = modelData
        self.actualData = actualData
        self.maturitySet = maturitySet

    def buildSignal(self, errorData, shortW, longW):
        signals = errorData.rolling(shortW).mean() - errorData.rolling(longW).mean()
        return signals
    
    def buildSlopes(self):

        slopes = [x for x in combinations(self.maturitySet, 2)]
        modelSlopes = pd.DataFrame()
        actualSlopes = pd.DataFrame()
        
        for slope in slopes:
            modelSlopes[f'{slope[0]}s{slope[1]}s'] = self.modelData[slope[1]] - self.modelData[slope[0]]
            actualSlopes[f'{slope[0]}s{slope[1]}s'] = self.actualData[slope[1]] - self.actualData[slope[0]]

        return {'model':modelSlopes, 'actual': actualSlopes}
    
    def buildFlies(self):
        flies = [(i, j, k) for i, j, k in combinations(self.maturitySet, 3) if (j - i) == (k - j)]
        modelFlies = pd.DataFrame()
        actualFlies = pd.DataFrame()
        for fly in flies:
            modelFlies[f'{fly[0]}s{fly[1]}s{fly[2]}s'] = self.modelData[fly[0]] - 2 * self.modelData[fly[1]] + self.modelData[fly[2]]
            actualFlies[f'{fly[0]}s{fly[1]}s{fly[2]}s'] = self.actualData[fly[0]] - 2 * self.actualData[fly[1]] + self.actualData[fly[2]]
        return {'model': modelFlies, 'actual': actualFlies}

    def screener(self, model, actual, shortW, longW):
        
        mispricings = model - actual
        signals = self.buildSignal(mispricings, shortW = shortW, longW = longW)
        
        modelLevels = model.iloc[-1]
        actualLevels = actual.iloc[-1]
        currentMispricing = mispricings.iloc[-1]
        currentSignals = signals.iloc[-1]

        summaryDf = pd.DataFrame({
                                  'model': modelLevels,
                                  'actual': actualLevels,
                                  'error': currentMispricing,
                                  'signal': currentSignals})
        
        return summaryDf

    def outrightScreener(self, shortW = 5, longW = 40):
        
        return self.screener(model = self.modelData[self.maturitySet], 
                             actual = self.actualData[self.maturitySet],
                             shortW = shortW,
                             longW = longW)
    
    def slopeScreener(self, shortW = 5, longW = 40):
        slopeDict = self.buildSlopes()

        return self.screener(model = slopeDict['model'],
                             actual = slopeDict['actual'],
                             shortW = shortW,
                             longW = longW)
    
    def flyScreener(self, shortW = 5, longW = 40):
        flyDict = self.buildFlies()
        return self.screener(model = flyDict['model'],
                             actual = flyDict['actual'],
                             shortW = shortW,
                             longW = longW)

    

    

        