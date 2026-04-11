import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

class tradeScreener:

    def __init__(self, modelData, actualData, maturitySet,
                 modelData_fwd, actualData_fwd, maturitySet_fwd):
        self.modelData = modelData
        self.modelData.index = pd.to_datetime(self.modelData.index)
        self.actualData = actualData
        self.actualData.index = pd.to_datetime(self.actualData.index)
        self.maturitySet = maturitySet
        
        self.modelData_fwd = modelData_fwd
        for key, _ in self.modelData_fwd.items():
            modelData_fwd[key].index = pd.to_datetime(modelData_fwd[key].index) 
        
        self.actualData_fwd = actualData_fwd
        for key, _ in self.modelData_fwd.items():
            self.actualData_fwd[key].index = pd.to_datetime(self.actualData_fwd[key].index)

        self.maturitySet_fwd = maturitySet_fwd

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
        
        mispricings = actual - model
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

    def singleItemPerformance(self, modelSeries, actualSeries, 
                              startDt, endDt,
                              shortW, longW, standardW = 14):
        
        mispricingSeries = (actualSeries - modelSeries)[startDt:endDt]
        reversionSignal = self.buildSignal(mispricingSeries, shortW = shortW, longW = longW)
        position = pd.Series(0, index = mispricingSeries.index)
        thr = mispricingSeries.rolling(standardW).std()
        position.loc[(mispricingSeries < -thr) & (reversionSignal > 0)] = +1
        position.loc[(mispricingSeries > thr) & (reversionSignal < 0)] = -1

        trades = []
        current = None

        for t in position.index:
            pos = position.loc[t]

            if current is None:
                if pos != 0:
                    current = {
                        'entry_date': t,
                        'side': 'LONG' if pos == 1 else 'SHORT',
                        'entry_misp': mispricingSeries.loc[t],
                        'entry_signal': reversionSignal.loc[t]
                    }
            else:
                if pos == 0:
                    current.update({
                        'exit_date': t,
                        'exit_misp': mispricingSeries.loc[t],
                        'exit_signal': reversionSignal.loc[t]
                    })
                    trades.append(current)
                    current = None

        trades_df = pd.DataFrame(trades)
        trades_df = trades_df[['entry_date', 'exit_date', 'side', 'entry_misp', 'exit_misp', 'entry_signal', 'exit_signal']]
        trades_df['pnl'] = 2*((trades_df['side'] == 'LONG').astype(int) - 0.5) * (trades_df['exit_misp'] - trades_df['entry_misp'])
        trades_df['days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
        trades_df['hit'] = trades_df['pnl'] > 0
        return trades_df
    
    def allOutrightBacktest(self, startDt, endDt, shortW, longW, standardW = 14):
        results_dict = {}
        for maturity in self.maturitySet:
            modelSeries = self.modelData[maturity][startDt:endDt]
            actualSeries = self.actualData[maturity][startDt:endDt]
            performanceDf = self.singleItemPerformance(modelSeries=modelSeries,
                                                       actualSeries=actualSeries,
                                                       startDt = startDt, endDt= endDt,
                                                       shortW = shortW, longW = longW, standardW = standardW)
            results_dict[maturity] = performanceDf

        results_df = pd.DataFrame([(
            (key, 
             (df['pnl'] > 0).mean(), 
             df.loc[df['pnl'] > 0]['pnl'].mean() / abs(df.loc[df['pnl'] < 0]['pnl'].mean()) if pd.notna(df.loc[df['pnl'] > 0]['pnl'].mean()) and df.loc[df['pnl'] < 0]['pnl'].mean() != 0 else pd.NA,
             df['days'].mean(),
             df['days'].median(),
             results_dict[key].shape[0]
             )) 
             for key, df in results_dict.items()], 
             columns = ['maturity', 'hitrate', 'skew', 'avg days', 'median days','n_trades'])
        return results_df

    def allSlopesBacktest(self, startDt, endDt, shortW, longW, standardW = 14):
        slopeDict = self.buildSlopes()
        modelSlopes = slopeDict['model']
        actualSlopes = slopeDict['actual']
        results_dict = {}
        for targetSlope in modelSlopes.columns:
            modelSeries = modelSlopes[targetSlope][startDt:endDt]
            actualSeries = actualSlopes[targetSlope][startDt:endDt]
            
            performanceDf = self.singleItemPerformance(modelSeries=modelSeries,
                                                       actualSeries=actualSeries,
                                                       startDt = startDt, endDt= endDt,
                                                       shortW = shortW, longW = longW, standardW = standardW)
            results_dict[targetSlope] = performanceDf

        results_df = pd.DataFrame([(
            (key, 
             (df['pnl'] > 0).mean(), 
             df.loc[df['pnl'] > 0]['pnl'].mean() / abs(df.loc[df['pnl'] < 0]['pnl'].mean()) if pd.notna(df.loc[df['pnl'] > 0]['pnl'].mean()) and df.loc[df['pnl'] < 0]['pnl'].mean() != 0 else pd.NA,
             df['days'].mean(),
             df['days'].median(),
             results_dict[key].shape[0]
             )) 
             for key, df in results_dict.items()], 
             columns = ['slope', 'hitrate', 'skew', 'avg days', 'median days','n_trades'])

        return results_df

    def allFliesBacktest(self, startDt, endDt, shortW, longW, standardW = 14):
        flies = [(i, j, k) for i, j, k in combinations(self.maturitySet, 3) if (j - i) == (k - j)]
        flyDict = self.buildFlies()
        modelFlies = flyDict['model']
        actualFlies = flyDict['actual']
        results_dict = {}
        for targetFly in flies:
            targetFly = f'{targetFly[0]}s{targetFly[1]}s{targetFly[2]}s'
            modelSeries = modelFlies[targetFly][startDt:endDt]
            actualSeries = actualFlies[targetFly][startDt:endDt]
            
            performanceDf = self.singleItemPerformance(modelSeries=modelSeries,
                                                       actualSeries=actualSeries,
                                                       startDt = startDt, endDt= endDt,
                                                       shortW = shortW, longW = longW, standardW = standardW)
            results_dict[targetFly] = performanceDf

        results_df = pd.DataFrame([(
            (key, 
             (df['pnl'] > 0).mean(), 
             df.loc[df['pnl'] > 0]['pnl'].mean() / abs(df.loc[df['pnl'] < 0]['pnl'].mean()) if pd.notna(df.loc[df['pnl'] > 0]['pnl'].mean()) and df.loc[df['pnl'] < 0]['pnl'].mean() != 0 else pd.NA,
             df['days'].mean(),
             df['days'].median(),
             results_dict[key].shape[0]
             )) 
             for key, df in results_dict.items()], 
             columns = ['fly', 'hitrate', 'skew', 'avg days', 'median days','n_trades'])

        return results_df

    def plotModelVsActual(self, modelSeries, actualSeries, 
                          startDt, endDt, leftPlotBp = False,
                          display_startDt = None, display_endDt = None,
                          shortW = 5, longW = 40, standardW = 14):
        
        actualSeries = actualSeries[startDt:endDt]
        modelSeries = modelSeries[startDt:endDt]
        mispricingSeries = actualSeries - modelSeries
        signalSeries = self.buildSignal(errorData=mispricingSeries, shortW=shortW, longW=longW)
        shortW_MA = mispricingSeries.rolling(shortW).mean() 
        longW_MA = mispricingSeries.rolling(longW).mean()

        if display_startDt or display_endDt:
            actualSeries = actualSeries[display_startDt:display_endDt]
            modelSeries = modelSeries[display_startDt:display_endDt]
            mispricingSeries = mispricingSeries[display_startDt:display_endDt]
            signalSeries = signalSeries[display_startDt:display_endDt]
            shortW_MA = shortW_MA[display_startDt:display_endDt]
            longW_MA = longW_MA[display_startDt:display_endDt]

        fig, ax = plt.subplots(nrows= 1, ncols = 2, figsize = (20, 4))
        ax[0].plot(modelSeries, color = 'blue', linestyle = '--', label = 'model')
        ax[0].plot(actualSeries, color = 'red', linestyle = '-', label = 'actual')
        ax[0].legend()
        ax[0].set_title('Level of actual versus model')
        if leftPlotBp:
            ax[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y*100:.1f} bp'))
        else:
            ax[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.2f}%'))
        x = modelSeries.index

        ax[1].plot(mispricingSeries, color= 'black', label = 'mispricing')
        ax[1].plot(longW_MA, color = 'blue', linestyle = '-', label = f'{longW}d MA', alpha = 1)
        ax[1].plot(shortW_MA, color = 'red', linestyle = '--', label = f'{shortW}d MA', alpha = 1)
        ax[1].legend()
        ax[1].set_title('Fitting error and reversion')
        ax[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y*100:.1f} bp'))
        for i in [0,1]:
            ax[i].fill_between(
                x, ax[i].get_ylim()[0], ax[i].get_ylim()[1],
                where=(signalSeries > 0 + signalSeries.rolling(standardW).std()),
                color='green', alpha=0.1, interpolate=True
            )

            ax[i].fill_between(
                x, ax[i].get_ylim()[0], ax[i].get_ylim()[1],
                where=(signalSeries < 0 - signalSeries.rolling(standardW).std()),
                color='red', alpha=0.1, interpolate=True
            )
            ax[i].grid(True)
        
    def outrightScreener_fwd(self, shortW = 5, longW= 40):
        
        deltaTauList = [x for x in self.modelData_fwd.keys()]
        outputdf = pd.DataFrame()
        for deltaTau in deltaTauList:
            summaryDf = self.screener(model = self.modelData_fwd[deltaTau], 
                                        actual = self.actualData_fwd[deltaTau], 
                                        shortW=shortW, longW=longW)
            summaryDf.index = [f'{deltaTau}y{x}y' for x in summaryDf.index]
            outputdf = pd.concat([outputdf, summaryDf])
        #outputdf = outputdf.style.format("{:.6f}")
        return outputdf.round(6)


    

        