import QuantLib as ql
import numpy as np
import pandas as pd

def _ql_date(ts):
    ts = pd.Timestamp(ts)
    return ql.Date(ts.day, ts.month, ts.year)

def _parse_trade_name(name):
    parts = str(name).split('s')
    nums = [float(x) for x in parts if x != '']
    if len(nums) == 1:
        return ('outright', nums)
    if len(nums) == 2:
        return ('slope', nums)
    if len(nums) == 3:
        return ('fly', nums)
    raise ValueError(f"Cannot parse trade name: {name}")

class swapEngine():

    def __init__(self, soniaFixings):
        self.soniaFixings = None
        if soniaFixings is not None:
            s = soniaFixings.copy()
            if isinstance(s, pd.DataFrame):
                if s.shape[1] != 1:
                    raise ValueError("soniaFixings DataFrame must have exactly one column")
                s = s.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            self.soniaFixings = s.sort_index()

    def _add_fixings(self, index, up_to_ts=None):
        if self.soniaFixings is None:
            return

        fixings = self.soniaFixings
        if up_to_ts is not None:
            fixings = fixings.loc[:pd.Timestamp(up_to_ts)]

        for dt, rate in fixings.items():
            qld = _ql_date(dt)
            if not index.hasHistoricalFixing(qld):
                index.addFixing(qld, float(rate) / 100.0)

    def make_zero_curve_from_row(self, row, eval_date, calendar=ql.UnitedKingdom()):
        row = row.copy()

        short_rate = float(row.pop('short')) / 100.0
        maturities = [float(c) for c in row.index]
        zero_rates = [float(x) / 100.0 for x in row.values]

        dates = [eval_date]
        rates = [short_rate]

        for t, r in zip(maturities, zero_rates):
            d = eval_date + ql.Period(int(round(t * 12)), ql.Months)
            dates.append(d)
            rates.append(r)

        curve = ql.ZeroCurve(
            dates,
            rates,
            ql.Actual365Fixed(),
            calendar,
            ql.Linear(),
            ql.Continuous,
            ql.Annual
        )
        handle = ql.YieldTermStructureHandle(curve)
        handle.enableExtrapolation()
        return handle

    def _make_ois_swap(self, eval_ts, maturity_years, fixed_rate, curve_handle,
                       notional=1_000_000):
        eval_date = _ql_date(eval_ts)
        ql.Settings.instance().evaluationDate = eval_date

        index = ql.Sonia(curve_handle)
        self._add_fixings(index, up_to_ts=eval_ts)
        tenor = ql.Period(int(round(maturity_years * 12)), ql.Months)

        swap = ql.MakeOIS(
            tenor,
            index,
            fixedRate=fixed_rate / 100.0,
            nominal=notional,
            settlementDays=0,
            paymentLag=0,
            fixedLegDayCount=ql.Actual365Fixed(),
            telescopicValueDates=True
        )
        swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
        return swap
    
    def _build_exact_ois_swap(self, entry_ts, maturity_years, fixed_rate, curve_handle,
                          notional=1_000_000, side=1):
        entry_date = _ql_date(entry_ts)
        ql.Settings.instance().evaluationDate = entry_date

        calendar = ql.UnitedKingdom()
        index = ql.Sonia(curve_handle)
        self._add_fixings(index, up_to_ts=entry_ts)

        effective_date = entry_date
        maturity_date = calendar.advance(
            effective_date,
            ql.Period(int(round(maturity_years * 12)), ql.Months)
        )

        fixed_schedule = ql.Schedule(
            effective_date,
            maturity_date,
            ql.Period(ql.Annual),
            calendar,
            ql.ModifiedFollowing,
            ql.ModifiedFollowing,
            ql.DateGeneration.Forward,
            False
        )

        # payer=Pay fixed / receiver=Receive fixed
        typ = ql.OvernightIndexedSwap.Receiver if side == 1 else ql.OvernightIndexedSwap.Payer

        swap = ql.OvernightIndexedSwap(
            typ,
            notional,
            fixed_schedule,
            fixed_rate / 100.0,
            ql.Actual365Fixed(),
            index
        )
        swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))

        return {
            "swap": swap,
            "type": typ,
            "notional": notional,
            "fixed_rate": fixed_rate,
            "effective_date": effective_date,
            "maturity_date": maturity_date,
            "fixed_schedule": fixed_schedule,
        }
    
    def _reprice_exact_ois_swap(self, trade_leg, val_ts, curve_handle):
        val_date = _ql_date(val_ts)
        ql.Settings.instance().evaluationDate = val_date

        if val_date >= trade_leg["maturity_date"]:
            return 0.0

        index = ql.Sonia(curve_handle)
        self._add_fixings(index, up_to_ts=val_ts)

        swap = ql.OvernightIndexedSwap(
            trade_leg["type"],
            trade_leg["notional"],
            trade_leg["fixed_schedule"],
            trade_leg["fixed_rate"] / 100.0,
            ql.Actual365Fixed(),
            index
        )
        swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
        return swap.NPV()
        
    def _par_rate_from_curve(self, eval_ts, maturity_years, curve_handle):
        tmp = self._make_ois_swap(
            eval_ts=eval_ts,
            maturity_years=maturity_years,
            fixed_rate=0.0,
            curve_handle=curve_handle,
            notional=1_000_000
        )
        return 100.0 * tmp.fairRate()

    def _dv01(self, eval_ts, maturity_years, curve_handle, notional=1_000_000):
        par = self._par_rate_from_curve(eval_ts, maturity_years, curve_handle)
        base = self._make_ois_swap(eval_ts, maturity_years, par, curve_handle, notional)
        npv0 = base.NPV()
        

        bumped_curve = ql.ZeroSpreadedTermStructure(
            curve_handle,
            ql.QuoteHandle(ql.SimpleQuote(1e-4))
        )
        bumped_handle = ql.YieldTermStructureHandle(bumped_curve)
        bumped_handle.enableExtrapolation()

        bumped = self._make_ois_swap(eval_ts, maturity_years, par, bumped_handle, notional)
        npv1 = bumped.NPV()

        return npv1 - npv0
    
    def _entry_weights(self, trade_name, entry_ts, zero_curve_row):
        trade_type, mats = _parse_trade_name(trade_name)
        curve_handle = self.make_zero_curve_from_row(zero_curve_row, _ql_date(entry_ts))

        if trade_type == 'outright':
            return {mats[0]: 1.0}

        if trade_type == 'slope':
            m1, m2 = mats
            dv01_1 = abs(self._dv01(entry_ts, m1, curve_handle))
            dv01_2 = abs(self._dv01(entry_ts, m2, curve_handle))
            return {m1: 1.0 / dv01_1, m2: -1.0 / dv01_2}

        if trade_type == 'fly':
            m1, m2, m3 = mats
            dv01_1 = abs(self._dv01(entry_ts, m1, curve_handle))
            dv01_2 = abs(self._dv01(entry_ts, m2, curve_handle))
            dv01_3 = abs(self._dv01(entry_ts, m3, curve_handle))
            #simple DV01-scaled 1:-2:1 fly
            return {
                m1: 1.0 / dv01_1,
                m2: -2.0 / dv01_2,
                m3: 1.0 / dv01_3
            }

        raise ValueError(f"Unsupported trade type: {trade_type}")
    
    def _build_trade_book(self, trade_name, side, entry_ts, zero_curve_row, par_curve_row,
                          base_notional=1_000_000):
        weights = self._entry_weights(trade_name = trade_name, 
                                      entry_ts = entry_ts, 
                                      zero_curve_row = zero_curve_row)

        trade_book = []
        trade_sign = 1 if side == 'LONG' else -1
        curve_handle = self.make_zero_curve_from_row(zero_curve_row, _ql_date(entry_ts))

        for mat, w in weights.items():
                fixed_rate = float(par_curve_row[mat])
                notional = w * base_notional
                ql_side = trade_sign if notional > 0 else -trade_sign

                leg = self._build_exact_ois_swap(
                    entry_ts=entry_ts,
                    maturity_years=mat,
                    fixed_rate=fixed_rate,
                    curve_handle=curve_handle,
                    notional=abs(notional),
                    side=ql_side
                )

                leg["entry_npv"] = leg["swap"].NPV()
                leg["maturity"] = mat
                trade_book.append(leg)

        return trade_book
    
    def _mark_trade_book(self, trade_book, val_ts, zero_curve_row):
        curve_handle = self.make_zero_curve_from_row(zero_curve_row, _ql_date(val_ts))
        total_npv = 0.0

        for leg in trade_book:
            total_npv += self._reprice_exact_ois_swap(leg, val_ts, curve_handle)

        return total_npv
