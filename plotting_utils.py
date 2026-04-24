import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as mtick

def residuals_actualFitted(fittingErrors, fittedTs, termStructurePath, tenor, sampleDates,
                           figsize = (20,5)):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))
    ax[0].plot(fittingErrors[tenor], color = 'blue')
    ax[1].plot(fittedTs[tenor], label = 'fitted', color = 'blue')
    ax[1].plot(termStructurePath[tenor], label = 'actual', color = 'red')

    ax[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    ax[0].set_ylabel('bps')
    ax[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y/100:.2%}'))
    ax[1].legend()
    ax[0].set_title(f'fitting errors on {tenor}y spot')
    ax[1].set_title(f'actual versus fitted {tenor}y yields')
    ax[0].axvline(x = pd.Timestamp(sampleDates[1]), color = 'grey', linestyle = '--')
    ax[1].axvline(x = pd.Timestamp(sampleDates[1]), color = 'grey', linestyle = '--')

def multipleResiduals(fittingErrors, tenors, sampleDates, figsize = (20,5)):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize)
    for t in tenors:
        ax.plot(fittingErrors[t], label = f'{t}y residual')

    ax.legend()
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    ax.set_ylabel('bps')
    ax.axvline(x = pd.Timestamp(sampleDates[1]), color = 'grey', linestyle = '--')

def threeFactorPlot(estFactorsDf_full, figsize = (20,3)):
    fig, ax = plt.subplots(figsize = figsize)
    plt.subplots_adjust(bottom=0.25)
    ax.plot(estFactorsDf_full['short'], color = 'blue')
    ax.plot(estFactorsDf_full['medium'], color = 'red', label = 'medium')
    ax.plot(estFactorsDf_full['long'], color = 'orange', label = 'long')
    ax.axhline(y = 0, color = 'grey', linestyle = '--')
    ax.legend()
    ax.set_title('Evolution of latent factors')

def twoFactorPlot(estFactorsDf_full, limits, figsize = (20,3)):
    
    fig, ax = plt.subplots(nrows= 1, ncols = 2, figsize = figsize)
    ax[0].plot(estFactorsDf_full['medium'][limits[0]:limits[1]], color = 'blue')
    ax[0].set_title('Medium factor')
    ax[1].plot(estFactorsDf_full['long'][limits[0]:limits[1]], color = 'blue')
    ax[1].set_title('Long factor')

def factorsForwardsPlot(estFactorsDf_full, sampleDates, figsize = (20,6)):
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = figsize)
    ax[0].plot(estFactorsDf_full['medium'], color = 'blue', linestyle = '-.', label = 'factor')
    ax[0].plot(estFactorsDf_full['2y1y'], color = 'blue', linestyle = '-', label = '2y1y')
    ax[0].axvline(x = pd.Timestamp(sampleDates[1]), color = 'grey', linestyle = '--')
    ax[0].set_title('Medium factor versus 2y-forward 1y')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(estFactorsDf_full['long'], color = 'red', linestyle = '-.', label = 'factor')
    ax[1].plot(estFactorsDf_full['10y1y'], color = 'red', linestyle = '-', label = '10y1y')
    ax[1].axvline(x = pd.Timestamp(sampleDates[1]), color = 'grey', linestyle = '--')
    ax[1].set_title('Long factor versus 10y-forward 1y')
    ax[1].grid(True)
    ax[1].legend()

def fittingErrorsHeatmap(fittingErrors_byYear, figsize = (12, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.Blues
    im = ax.imshow(fittingErrors_byYear.values, aspect='auto', cmap = cmap)

    ax.set_xticks(np.arange(len(fittingErrors_byYear.columns)))
    ax.set_yticks(np.arange(len(fittingErrors_byYear.index)))
    ax.set_xticklabels(fittingErrors_byYear.columns)
    ax.set_yticklabels(fittingErrors_byYear.index)
    ax.set_xlabel("Year")
    ax.set_ylabel("Maturity")
    ax.set_title("Root Mean Squared Fitting Errors (bps)")

    norm = im.norm
    threshold = (norm.vmax + norm.vmin) / 2

    # annotate cells
    for i in range(fittingErrors_byYear.shape[0]):
        for j in range(fittingErrors_byYear.shape[1]):
            value = fittingErrors_byYear.iloc[i, j]
            color = "white" if norm(value) > norm(threshold) else "black"
            ax.text(j, i, f"{value:.1f}",
                    ha="center", va="center", fontsize=8, color=color)

    # colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("bps")

    plt.tight_layout()
    plt.show()

def actualVsFittedCurve(targetDate, fittedTs_full_df, termStructurePath, 
                        tenorsAbove= 4, figsize = (20,3)):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
    fittedCurve = fittedTs_full_df.loc[targetDate]
    actualCurve =termStructurePath.drop('short', axis = 1).loc[targetDate]

    ax[0].plot(fittedCurve, color = 'blue', label = 'fitted')
    ax[0].plot(actualCurve, color = 'red', label = 'actual')
    ax[0].legend()
    ax[0].set_title(f'Model versus actual zero curve on {targetDate}')

    selectedTenors = [x for x in termStructurePath.drop('short', axis = 1).columns if int(x) > tenorsAbove]
    selectedMispr = (fittedCurve[selectedTenors] - actualCurve[selectedTenors]).values
    ax[1].bar(selectedTenors, selectedMispr, color = 'purple')
    ax[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y*100:.0f} bps'))
    ax[1].set_title(f'Fitted yield - Actual yield on {targetDate}')

def currentErrorsHeatmap_fwd(fittedForwardTs_full, forwardTermStructurePath, figsize = (12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    errorData = pd.DataFrame({x: 
              100 *(fittedForwardTs_full[x].iloc[-1] - forwardTermStructurePath[x].iloc[-1]) 
              for x in fittedForwardTs_full.keys()}).round(6)
    cmap = plt.cm.Blues
    im = ax.imshow(errorData.values, aspect='auto', cmap = cmap)

    ax.set_xticks(np.arange(len(errorData.columns)))
    ax.set_yticks(np.arange(len(errorData.index)))
    ax.set_xticklabels(errorData.columns)
    ax.set_yticklabels(errorData.index)
    ax.set_ylabel("Maturity")
    ax.set_xlabel("Years Forward")
    ax.set_title(f"Errors of fit on forwards as of {fittedForwardTs_full[1].iloc[-1].name.strftime('%Y-%m-%d')}")

    norm = im.norm
    threshold = (norm.vmax + norm.vmin) / 2

    # annotate cells
    for i in range(errorData.shape[0]):
        for j in range(errorData.shape[1]):
            value = errorData.iloc[i, j]
            if pd.isna(value):
                continue
            else:
                color = "white" if norm(value) > norm(threshold) else "black"
            ax.text(j, i, f"{value:.1f}",
                        ha="center", va="center", fontsize=8, color=color)

    # colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("bps")

    plt.tight_layout()
    plt.show()

def fittingErrorsHeatmap_fwd(rmse_by_x, x_list, figsize=(20, 5)):
    
    fig, axes = plt.subplots(1, len(x_list), figsize=figsize, sharey=True)

    cmap = plt.cm.Blues
    vmin = min(rmse_by_x[x].min().min() for x in x_list)
    vmax = max(rmse_by_x[x].max().max() for x in x_list)

    for ax, x in zip(axes, x_list):
        fittingErrors_byYear = rmse_by_x[x]

        im = ax.imshow(fittingErrors_byYear.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(fittingErrors_byYear.columns)))
        ax.set_yticks(np.arange(len(fittingErrors_byYear.index)))
        ax.set_xticklabels(fittingErrors_byYear.columns)
        ax.set_yticklabels(fittingErrors_byYear.index)
        ax.set_xlabel("Year")
        ax.set_title(f"{x}y fwd")

        norm = im.norm
        threshold = (norm.vmax + norm.vmin) / 2

        # annotate
        for i in range(fittingErrors_byYear.shape[0]):
            for j in range(fittingErrors_byYear.shape[1]):
                if pd.isna(fittingErrors_byYear.iloc[i, j]):
                    continue
                else:
                    value = fittingErrors_byYear.iloc[i, j]
                    color = "white" if norm(value) > norm(threshold) else "black"
                    ax.text(j, i, f"{value:.1f}",
                            ha="center", va="center", fontsize=6, color=color)

    axes[0].set_ylabel("Maturity")

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.01)
    cbar.set_label("bps")

    plt.suptitle("Root Mean Squared Fitting Errors (bps)")
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plt.show()

