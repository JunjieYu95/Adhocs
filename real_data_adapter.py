"""
Real Data Adapter for Deep Hedging
----------------------------------
Transform real market data into the format expected by VanillaDeepHedgingGym.

This module provides functionality to convert real option market data (strikes, prices, 
implied volatilities, etc.) into the structured format required by the deep hedging framework.

Author: AI Assistant
Created: 2024
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from scipy.interpolate import interp1d
import warnings
from typing import Dict, Optional, Union, List, Tuple, Callable
from collections.abc import Mapping

from .base import Logger, Config, dh_dtype, tf, tfCast, pdct, tf_dict, assert_iter_not_is_nan, DIM_DUMMY
from .world import SimpleWorld_Spot_ATM

_log = Logger(__file__)

class RealDataAdapter:
    """
    Adapter to transform real market data into Deep Hedging world format.
    
    This class takes real option market data and converts it into the structured
    format expected by VanillaDeepHedgingGym, handling missing data, interpolation,
    and proper calculation of hedge returns.
    
    Attributes
    ----------
    nSamples : int
        Number of samples/paths
    nSteps : int  
        Number of time steps
    nInst : int
        Number of instruments (typically 2: spot + option)
    dt : float
        Time step size
    tf_dtype : tf.DType
        TensorFlow data type
    np_dtype : numpy.dtype
        NumPy data type
    """
    
    def __init__(self, 
                 market_data: Union[pd.DataFrame, Dict],
                 config: Optional[Config] = None,
                 dtype: tf.DType = dh_dtype):
        """
        Initialize the adapter with real market data.
        
        Parameters
        ----------
        market_data : pd.DataFrame or dict
            Real market data containing at minimum:
            - 'strike': Strike prices
            - 'price': Option prices  
            - 'ivol': Implied volatilities
            - 'time_left': Time to expiration
            Optional columns:
            - 'spot': Underlying spot price
            - 'delta': Option delta
            - 'vega': Option vega
            - 'timestamp': Time stamps
            
        config : Config, optional
            Configuration parameters. If None, uses defaults.
            
        dtype : tf.DType, optional
            Data type for tensors
        """
        self.tf_dtype = dtype
        self.np_dtype = dtype.as_numpy_dtype()
        
        # Setup default config if none provided
        if config is None:
            config = Config()
            
        # Parse configuration
        self._parse_config(config)
        
        # Process market data
        self.raw_data = self._validate_and_process_data(market_data)
        
        # Initialize dimensions based on data
        self._initialize_dimensions()
        
        # Build world data structure
        self.data = None
        self.tf_data = None
        self._build_world_data()
        
    def _parse_config(self, config: Config):
        """Parse configuration parameters."""
        # Market parameters
        self.spot_initial = config("spot_initial", 1.0, float, "Initial spot price if not in data")
        self.cost_s = config("cost_s", 0.0002, float, "Trading cost for spot")
        self.cost_v = config("cost_v", 0.02, float, "Trading cost for vega")
        self.cost_p = config("cost_p", 0.0005, float, "Additional option trading cost")
        
        # Action bounds
        self.ubnd_as = config("ubnd_as", 5.0, float, "Upper bound for spot actions")
        self.lbnd_as = config("lbnd_as", -5.0, float, "Lower bound for spot actions")
        self.ubnd_av = config("ubnd_av", 5.0, float, "Upper bound for option actions")
        self.lbnd_av = config("lbnd_av", -5.0, float, "Lower bound for option actions")
        
        # Data processing
        self.max_time_steps = config("max_time_steps", 21, int, "Maximum number of time steps")
        self.time_step_size = config("time_step_size", 1.0, float, "Time step size in same units as time_left")
        self.interpolate_missing = config("interpolate_missing", True, bool, "Whether to interpolate missing data")
        
        # Payoff definition
        payoff_type = config("payoff_type", "short_call", str, "Type of payoff: 'short_call', 'short_put', 'custom'")
        if payoff_type == "short_call":
            self.payoff_func = lambda spots, strikes: -np.maximum(spots[:, -1:] - strikes, 0.0)
        elif payoff_type == "short_put":
            self.payoff_func = lambda spots, strikes: -np.maximum(strikes - spots[:, -1:], 0.0)
        else:
            # Allow custom payoff function
            self.payoff_func = config("payoff_func", self._default_payoff, help="Custom payoff function")
            
        config.done()
        
    def _default_payoff(self, spots: np.ndarray, strikes: np.ndarray) -> np.ndarray:
        """Default payoff function (short call)."""
        return -np.maximum(spots[:, -1:] - strikes, 0.0)
        
    def _validate_and_process_data(self, market_data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Validate and process the input market data."""
        if isinstance(market_data, dict):
            market_data = pd.DataFrame(market_data)
        elif not isinstance(market_data, pd.DataFrame):
            raise ValueError("market_data must be a pandas DataFrame or dictionary")
            
        # Check required columns
        required_cols = ['strike', 'price', 'ivol', 'time_left']
        missing_cols = [col for col in required_cols if col not in market_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Sort by time_left (descending) and strike
        market_data = market_data.sort_values(['time_left', 'strike'], ascending=[False, True])
        
        # Add spot if not present (assume ATM for first strike)
        if 'spot' not in market_data.columns:
            # Use first strike as proxy for spot
            representative_strike = market_data['strike'].iloc[0]
            market_data['spot'] = representative_strike
            _log.warning("'spot' column not found. Using first strike %g as proxy", representative_strike)
            
        # Calculate missing Greeks if not present
        if 'delta' not in market_data.columns:
            market_data['delta'] = self._calculate_bs_delta(
                market_data['spot'], market_data['strike'], 
                market_data['time_left'], market_data['ivol']
            )
            
        if 'vega' not in market_data.columns:
            market_data['vega'] = self._calculate_bs_vega(
                market_data['spot'], market_data['strike'],
                market_data['time_left'], market_data['ivol']
            )
            
        return market_data
        
    def _calculate_bs_delta(self, spot: float, strike: float, time_left: float, ivol: float) -> float:
        """Calculate Black-Scholes delta."""
        if time_left <= 0:
            return (spot > strike).astype(float)
            
        d1 = (np.log(spot / strike) + 0.5 * ivol**2 * time_left) / (ivol * np.sqrt(time_left))
        return norm.cdf(d1)
        
    def _calculate_bs_vega(self, spot: float, strike: float, time_left: float, ivol: float) -> float:
        """Calculate Black-Scholes vega."""
        if time_left <= 0:
            return 0.0
            
        d1 = (np.log(spot / strike) + 0.5 * ivol**2 * time_left) / (ivol * np.sqrt(time_left))
        return spot * norm.pdf(d1) * np.sqrt(time_left)
        
    def _initialize_dimensions(self):
        """Initialize dimensions based on the data."""
        # Determine number of time steps
        unique_times = sorted(self.raw_data['time_left'].unique(), reverse=True)
        self.nSteps = min(len(unique_times), self.max_time_steps)
        
        # Group data by time_left and count unique strikes at each time
        strikes_per_time = self.raw_data.groupby('time_left')['strike'].nunique()
        self.nSamples = strikes_per_time.iloc[0] if len(strikes_per_time) > 0 else len(self.raw_data)
        
        # Number of instruments (spot + option)
        self.nInst = 2
        
        # Time step
        if len(unique_times) > 1:
            self.dt = unique_times[0] - unique_times[1] 
        else:
            self.dt = self.time_step_size
            
        _log.info("Initialized dimensions: nSamples=%d, nSteps=%d, nInst=%d, dt=%g", 
                 self.nSamples, self.nSteps, self.nInst, self.dt)
                 
    def _build_world_data(self):
        """Build the world data structure expected by the gym."""
        # Organize data by time steps
        time_steps = sorted(self.raw_data['time_left'].unique(), reverse=True)[:self.nSteps]
        
        # Initialize arrays
        spot_path = np.zeros((self.nSamples, self.nSteps + 1), dtype=self.np_dtype)
        option_prices = np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype)
        option_deltas = np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype)
        option_vegas = np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype)
        strikes = np.zeros((self.nSamples,), dtype=self.np_dtype)
        ivols = np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype)
        
        # Process each time step
        for t_idx, time_left in enumerate(time_steps):
            time_data = self.raw_data[self.raw_data['time_left'] == time_left]
            
            # Ensure we have data for each sample
            time_data = time_data.head(self.nSamples)
            
            if len(time_data) < self.nSamples:
                # Interpolate or repeat data if needed
                time_data = self._fill_missing_samples(time_data, self.nSamples)
                
            # Fill arrays
            if t_idx == 0:  # First time step
                spot_path[:, t_idx] = time_data['spot'].values
                strikes[:] = time_data['strike'].values
                
            option_prices[:, t_idx] = time_data['price'].values
            option_deltas[:, t_idx] = time_data['delta'].values  
            option_vegas[:, t_idx] = time_data['vega'].values
            ivols[:, t_idx] = time_data['ivol'].values
            
        # Generate spot path (simplified - constant for now)
        for t in range(1, self.nSteps + 1):
            spot_path[:, t] = spot_path[:, 0]  # Can be enhanced with spot evolution
            
        # Calculate option payoffs at maturity
        option_payoffs = self.payoff_func(spot_path, strikes[:, np.newaxis])
        
        # Calculate hedge returns (dH_t = H_T - H_t)
        spot_returns = spot_path[:, self.nSteps:self.nSteps+1] - spot_path[:, :self.nSteps]
        option_returns = option_payoffs[:, np.newaxis] - option_prices
        
        # Combine hedge instruments
        hedges = np.zeros((self.nSamples, self.nSteps, self.nInst), dtype=self.np_dtype)
        hedges[:, :, 0] = spot_returns  # Spot hedges
        hedges[:, :, 1] = option_returns  # Option hedges
        
        # Calculate trading costs
        cost_spot = spot_path[:, :self.nSteps] * self.cost_s
        cost_option = (self.cost_v * np.abs(option_vegas) + 
                      self.cost_s * np.abs(option_deltas) + 
                      self.cost_p * np.abs(option_prices))
        
        cost = np.zeros((self.nSamples, self.nSteps, self.nInst), dtype=self.np_dtype)
        cost[:, :, 0] = cost_spot
        cost[:, :, 1] = cost_option
        
        # Action bounds
        ubnd_a = np.zeros((self.nSamples, self.nSteps, self.nInst), dtype=self.np_dtype)
        lbnd_a = np.zeros((self.nSamples, self.nSteps, self.nInst), dtype=self.np_dtype)
        ubnd_a[:, :, 0] = self.ubnd_as
        ubnd_a[:, :, 1] = self.ubnd_av
        lbnd_a[:, :, 0] = self.lbnd_as
        lbnd_a[:, :, 1] = self.lbnd_av
        
        # Terminal payoff
        payoff = option_payoffs.flatten()
        
        # Time features
        time_left_grid = np.array([time_steps[i] if i < len(time_steps) else 0.0 
                                  for i in range(self.nSteps)])
        time_left_features = np.tile(time_left_grid[np.newaxis, :], (self.nSamples, 1))
        sqrt_time_left_features = np.sqrt(np.maximum(time_left_features, 0.0))
        
        # Prices for features
        prices = np.zeros((self.nSamples, self.nSteps, self.nInst), dtype=self.np_dtype)
        prices[:, :, 0] = spot_path[:, :self.nSteps]
        prices[:, :, 1] = option_prices
        
        # Build data structure
        self.data = pdct()
        self.data.market = pdct(
            hedges=hedges,
            cost=cost,
            ubnd_a=ubnd_a,
            lbnd_a=lbnd_a,
            payoff=payoff
        )
        
        self.data.features = pdct(
            per_step=pdct(
                cost=cost,
                price=prices,
                ubnd_a=ubnd_a,
                lbnd_a=lbnd_a,
                time_left=time_left_features,
                sqrt_time_left=sqrt_time_left_features,
                spot=spot_path[:, :self.nSteps],
                ivol=ivols,
                call_price=option_prices,
                call_delta=option_deltas,
                call_vega=option_vegas,
                cost_v=cost[:, :, 1:2]
            ),
            per_path=pdct()
        )
        
        # Required dummy dimension
        self.data.features.per_path[DIM_DUMMY] = (payoff * 0.)[:, np.newaxis]
        
        # Check numerics
        assert_iter_not_is_nan(self.data, "data")
        
        # Convert to TensorFlow format
        self.tf_data = tf_dict(
            features=self.data.features,
            market=self.data.market,
            dtype=self.tf_dtype
        )
        
        # Additional attributes for compatibility
        self.sample_weights = np.full((self.nSamples, 1), 1.0/float(self.nSamples), dtype=self.np_dtype)
        self.tf_sample_weights = tf.constant(self.sample_weights, dtype=self.tf_dtype)
        self.sample_weights = self.sample_weights.reshape((self.nSamples,))
        self.tf_y = tf.zeros((self.nSamples,), dtype=self.tf_dtype)
        
        # Details for visualization
        self.details = pdct(
            spot_all=spot_path,
            strikes=strikes,
            option_prices=option_prices,
            option_deltas=option_deltas,
            option_vegas=option_vegas,
            ivols=ivols
        )
        
        # Timeline
        self.timeline = np.cumsum(np.linspace(0., self.nSteps, self.nSteps+1, 
                                            endpoint=True, dtype=np.float32)) * self.dt
        
        # Instrument names
        self.inst_names = ['spot', 'option']
        
    def _fill_missing_samples(self, time_data: pd.DataFrame, target_samples: int) -> pd.DataFrame:
        """Fill missing samples by interpolation or repetition."""
        if len(time_data) >= target_samples:
            return time_data.head(target_samples)
            
        if self.interpolate_missing and len(time_data) > 1:
            # Interpolate missing strikes
            existing_strikes = time_data['strike'].values
            min_strike, max_strike = existing_strikes.min(), existing_strikes.max()
            
            # Generate target strikes
            target_strikes = np.linspace(min_strike, max_strike, target_samples)
            
            # Interpolate each column
            interpolated_data = {}
            for col in time_data.columns:
                if col in ['strike']:
                    interpolated_data[col] = target_strikes
                else:
                    # Interpolate based on strike
                    f = interp1d(existing_strikes, time_data[col].values, 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
                    interpolated_data[col] = f(target_strikes)
                    
            return pd.DataFrame(interpolated_data)
        else:
            # Simply repeat the last row
            repeated_data = time_data.iloc[-1:].copy()
            for _ in range(target_samples - len(time_data)):
                time_data = pd.concat([time_data, repeated_data], ignore_index=True)
            return time_data.head(target_samples)
            
    def clone(self, **kwargs) -> 'RealDataAdapter':
        """Create a copy with modified parameters."""
        # Create new config with overrides
        config = Config()
        for key, value in kwargs.items():
            setattr(config, key, value)
            
        return RealDataAdapter(self.raw_data.copy(), config, self.tf_dtype)
        
    def get_market_summary(self) -> Dict:
        """Get a summary of the market data."""
        return {
            'nSamples': self.nSamples,
            'nSteps': self.nSteps, 
            'nInst': self.nInst,
            'dt': self.dt,
            'strike_range': (self.raw_data['strike'].min(), self.raw_data['strike'].max()),
            'time_range': (self.raw_data['time_left'].min(), self.raw_data['time_left'].max()),
            'price_range': (self.raw_data['price'].min(), self.raw_data['price'].max()),
            'ivol_range': (self.raw_data['ivol'].min(), self.raw_data['ivol'].max()),
            'available_features': sorted(self.data.features.per_step.keys()),
            'instruments': self.inst_names
        }
        
    @classmethod
    def from_csv(cls, 
                 csv_path: str,
                 config: Optional[Config] = None,
                 dtype: tf.DType = dh_dtype,
                 **pandas_kwargs) -> 'RealDataAdapter':
        """Create adapter from CSV file."""
        market_data = pd.read_csv(csv_path, **pandas_kwargs)
        return cls(market_data, config, dtype)
        
    @classmethod 
    def from_excel(cls,
                   excel_path: str,
                   config: Optional[Config] = None,
                   dtype: tf.DType = dh_dtype,
                   **pandas_kwargs) -> 'RealDataAdapter':
        """Create adapter from Excel file."""
        market_data = pd.read_excel(excel_path, **pandas_kwargs)
        return cls(market_data, config, dtype)


def test_trained_agent(gym, real_world: RealDataAdapter, verbose: bool = True) -> Dict:
    """
    Test a trained agent on real market data.
    
    Parameters
    ----------
    gym : VanillaDeepHedgingGym
        Trained gym containing the agent
    real_world : RealDataAdapter  
        Real market data adapter
    verbose : bool
        Whether to print results
        
    Returns
    -------
    dict
        Results dictionary with performance metrics
    """
    # Ensure gym is built
    if gym.agent is None:
        _ = gym(real_world.tf_data)
        
    # Run the trained agent on real data
    results = gym(real_world.tf_data)
    
    # Extract numpy results
    results_np = {key: val.numpy() if hasattr(val, 'numpy') else val 
                  for key, val in results.items()}
    
    # Calculate summary statistics
    weights = real_world.sample_weights
    summary = {
        'mean_utility': np.average(results_np['utility'], weights=weights),
        'mean_utility0': np.average(results_np['utility0'], weights=weights), 
        'mean_loss': np.average(results_np['loss'], weights=weights),
        'mean_pnl': np.average(results_np['pnl'], weights=weights),
        'mean_cost': np.average(results_np['cost'], weights=weights),
        'mean_gains': np.average(results_np['gains'], weights=weights),
        'std_gains': np.sqrt(np.average((results_np['gains'] - np.average(results_np['gains'], weights=weights))**2, weights=weights)),
        'min_gains': np.min(results_np['gains']),
        'max_gains': np.max(results_np['gains']),
        'market_summary': real_world.get_market_summary()
    }
    
    if verbose:
        print("\n" + "="*60)
        print("REAL MARKET DATA TEST RESULTS")
        print("="*60)
        print(f"Market Data Summary:")
        ms = summary['market_summary']
        print(f"  Samples: {ms['nSamples']}, Steps: {ms['nSteps']}, Instruments: {ms['nInst']}")
        print(f"  Strike range: [{ms['strike_range'][0]:.2f}, {ms['strike_range'][1]:.2f}]")
        print(f"  Time range: [{ms['time_range'][0]:.2f}, {ms['time_range'][1]:.2f}]")
        print(f"  Price range: [{ms['price_range'][0]:.4f}, {ms['price_range'][1]:.4f}]")
        print(f"  IV range: [{ms['ivol_range'][0]:.2%}, {ms['ivol_range'][1]:.2%}]")
        print()
        print("Performance Metrics:")
        print(f"  Mean Utility: {summary['mean_utility']:.6f}")
        print(f"  Mean Utility0: {summary['mean_utility0']:.6f}")  
        print(f"  Mean Loss: {summary['mean_loss']:.6f}")
        print(f"  Mean PnL: {summary['mean_pnl']:.6f}")
        print(f"  Mean Cost: {summary['mean_cost']:.6f}")
        print(f"  Mean Gains: {summary['mean_gains']:.6f}")
        print(f"  Std Gains: {summary['std_gains']:.6f}")
        print(f"  Min/Max Gains: [{summary['min_gains']:.6f}, {summary['max_gains']:.6f}]")
        print("="*60)
    
    # Add full results
    summary['full_results'] = results_np
    summary['real_world'] = real_world
    
    return summary


def load_sample_data() -> pd.DataFrame:
    """
    Create sample market data for testing purposes.
    This generates data similar to what might be in a real option chain.
    """
    np.random.seed(42)  # For reproducibility
    
    # Time grid
    times = [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    strikes = [55, 56, 57, 58, 59, 60, 61, 62]
    
    data_list = []
    spot = 57.0  # Base spot price
    
    for time_left in times:
        for strike in strikes:
            # Generate realistic option data
            # IV smile: higher vol for OTM options
            moneyness = strike / spot
            base_iv = 0.20 + 0.05 * abs(moneyness - 1.0)  # IV smile
            iv = base_iv + 0.01 * np.random.normal()  # Add noise
            iv = max(0.05, iv)  # Floor at 5%
            
            # Black-Scholes pricing
            time_to_exp = time_left / 365.0  # Convert to years
            if time_to_exp > 0:
                d1 = (np.log(spot / strike) + 0.5 * iv**2 * time_to_exp) / (iv * np.sqrt(time_to_exp))
                d2 = d1 - iv * np.sqrt(time_to_exp)
                call_price = spot * norm.cdf(d1) - strike * norm.cdf(d2)
                delta = norm.cdf(d1)
                vega = spot * norm.pdf(d1) * np.sqrt(time_to_exp)
            else:
                call_price = max(0, spot - strike)
                delta = 1.0 if spot > strike else 0.0
                vega = 0.0
                
            # Add market noise
            price_noise = 0.02 * call_price * np.random.normal()
            call_price = max(0.01, call_price + price_noise)
            
            data_list.append({
                'strike': strike,
                'price': call_price,
                'ivol': iv,
                'time_left': time_left,
                'spot': spot,
                'delta': delta, 
                'vega': vega
            })
    
    return pd.DataFrame(data_list)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_data = load_sample_data()
    print("Sample data shape:", sample_data.shape)
    print(sample_data.head())
    
    # Create adapter
    config = Config()
    adapter = RealDataAdapter(sample_data, config)
    
    # Print summary
    summary = adapter.get_market_summary()
    print("\nMarket Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
        
    print("\nAdapter created successfully!")
    print(f"tf_data keys: {list(adapter.tf_data.keys())}")
    print(f"market keys: {list(adapter.tf_data['market'].keys())}")
    print(f"features keys: {list(adapter.tf_data['features'].keys())}") 