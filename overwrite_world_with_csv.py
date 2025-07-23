#!/usr/bin/env python3
"""
Simple script to overwrite validation world with real CSV data
============================================================

This script takes a cloned validation world and overwrites its data
with real market data from a CSV file. Much simpler than creating
a custom adapter class.

Expected CSV format:
- strike: Strike prices
- price: Option prices  
- ivol: Implied volatilities (as decimals, e.g. 0.20 for 20%)
- time_left: Time to expiration
- spot: Underlying spot price (optional)

Author: AI Assistant  
Created: 2024
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from deephedging.base import tf_dict, DIM_DUMMY
import deephedging.base as base

def load_csv_data(csv_path):
    """Load and validate CSV data."""
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['strike', 'price', 'ivol', 'time_left']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Add spot if missing (use first strike as proxy)
    if 'spot' not in df.columns:
        df['spot'] = df['strike'].iloc[0]
        print(f"Warning: 'spot' column missing, using {df['spot'].iloc[0]} as spot price")
    
    # Calculate missing Greeks
    if 'delta' not in df.columns:
        df['delta'] = calculate_bs_delta(df['spot'], df['strike'], df['time_left'], df['ivol'])
    
    if 'vega' not in df.columns:
        df['vega'] = calculate_bs_vega(df['spot'], df['strike'], df['time_left'], df['ivol'])
    
    print(f"Loaded CSV data: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Time range: {df['time_left'].min()} to {df['time_left'].max()}")
    print(f"Strike range: {df['strike'].min()} to {df['strike'].max()}")
    
    return df

def calculate_bs_delta(spot, strike, time_left, ivol):
    """Calculate Black-Scholes delta."""
    time_to_exp = time_left / 365.0  # Assume time_left is in days
    
    # Handle zero time case
    mask_zero_time = time_to_exp <= 0
    delta = np.zeros_like(spot)
    
    # For zero time, delta is 1 if ITM, 0 if OTM
    delta[mask_zero_time] = (spot[mask_zero_time] > strike[mask_zero_time]).astype(float)
    
    # For positive time, use Black-Scholes formula
    mask_pos_time = time_to_exp > 0
    if np.any(mask_pos_time):
        d1 = (np.log(spot[mask_pos_time] / strike[mask_pos_time]) + 
              0.5 * ivol[mask_pos_time]**2 * time_to_exp[mask_pos_time]) / (
              ivol[mask_pos_time] * np.sqrt(time_to_exp[mask_pos_time]))
        delta[mask_pos_time] = norm.cdf(d1)
    
    return delta

def calculate_bs_vega(spot, strike, time_left, ivol):
    """Calculate Black-Scholes vega."""
    time_to_exp = time_left / 365.0
    
    # Handle zero time case
    vega = np.zeros_like(spot)
    mask_pos_time = time_to_exp > 0
    
    if np.any(mask_pos_time):
        d1 = (np.log(spot[mask_pos_time] / strike[mask_pos_time]) + 
              0.5 * ivol[mask_pos_time]**2 * time_to_exp[mask_pos_time]) / (
              ivol[mask_pos_time] * np.sqrt(time_to_exp[mask_pos_time]))
        vega[mask_pos_time] = spot[mask_pos_time] * norm.pdf(d1) * np.sqrt(time_to_exp[mask_pos_time])
    
    return vega

def overwrite_world_with_csv(val_world, csv_path, 
                           cost_s=0.0002, cost_v=0.02, cost_p=0.0005,
                           payoff_type="short_call"):
    """
    Overwrite validation world data with CSV data.
    
    Parameters
    ----------
    val_world : SimpleWorld_Spot_ATM
        Validation world created with world.clone()
    csv_path : str  
        Path to CSV file with market data
    cost_s : float
        Spot trading cost
    cost_v : float
        Vega trading cost  
    cost_p : float
        Additional option trading cost
    payoff_type : str
        "short_call" or "short_put"
    
    Returns
    -------
    Modified val_world with CSV data
    """
    
    # Load CSV data
    df = load_csv_data(csv_path)
    
    # Sort by time_left (descending) and strike
    df = df.sort_values(['time_left', 'strike'], ascending=[False, True])
    
    # Determine dimensions
    unique_times = sorted(df['time_left'].unique(), reverse=True)
    nSteps = len(unique_times)
    
    # Group by time and get number of strikes at each time
    strikes_per_time = df.groupby('time_left')['strike'].nunique()
    nSamples = strikes_per_time.iloc[0] if len(strikes_per_time) > 0 else len(df)
    nInst = 2  # spot + option
    
    print(f"New dimensions: nSamples={nSamples}, nSteps={nSteps}, nInst={nInst}")
    
    # Initialize arrays
    spot_path = np.zeros((nSamples, nSteps + 1), dtype=val_world.np_dtype)
    option_prices = np.zeros((nSamples, nSteps), dtype=val_world.np_dtype)
    option_deltas = np.zeros((nSamples, nSteps), dtype=val_world.np_dtype)
    option_vegas = np.zeros((nSamples, nSteps), dtype=val_world.np_dtype)
    strikes = np.zeros(nSamples, dtype=val_world.np_dtype)
    ivols = np.zeros((nSamples, nSteps), dtype=val_world.np_dtype)
    
    # Fill arrays with CSV data
    for t_idx, time_left in enumerate(unique_times):
        time_data = df[df['time_left'] == time_left].head(nSamples)
        
        # Pad with last row if needed
        while len(time_data) < nSamples:
            time_data = pd.concat([time_data, time_data.iloc[-1:]], ignore_index=True)
        
        if t_idx == 0:  # First time step
            spot_path[:, t_idx] = time_data['spot'].values[:nSamples]
            strikes[:] = time_data['strike'].values[:nSamples]
        
        option_prices[:, t_idx] = time_data['price'].values[:nSamples]
        option_deltas[:, t_idx] = time_data['delta'].values[:nSamples]
        option_vegas[:, t_idx] = time_data['vega'].values[:nSamples]
        ivols[:, t_idx] = time_data['ivol'].values[:nSamples]
    
    # Complete spot path (assume constant for simplicity)
    for t in range(1, nSteps + 1):
        spot_path[:, t] = spot_path[:, 0]
    
    # Calculate payoffs
    if payoff_type == "short_call":
        payoff = -np.maximum(spot_path[:, -1] - strikes, 0.0)
    elif payoff_type == "short_put":
        payoff = -np.maximum(strikes - spot_path[:, -1], 0.0)
    else:
        raise ValueError(f"Unknown payoff_type: {payoff_type}")
    
    # Calculate hedge returns (dH_t = H_T - H_t)
    # For spot: returns are the difference between final spot and current spot
    spot_returns = spot_path[:, nSteps][:, np.newaxis] - spot_path[:, :nSteps]
    
    # For options: returns are the difference between final payoff and current price
    option_returns = payoff[:, np.newaxis] - option_prices
    
    # Combine hedges
    hedges = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    hedges[:, :, 0] = spot_returns  # Spot
    hedges[:, :, 1] = option_returns  # Option
    
    # Calculate costs
    cost_spot = spot_path[:, :nSteps] * cost_s
    cost_option = (cost_v * np.abs(option_vegas) + 
                   cost_s * np.abs(option_deltas) + 
                   cost_p * np.abs(option_prices))
    
    cost = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    cost[:, :, 0] = cost_spot
    cost[:, :, 1] = cost_option
    
    # Action bounds (copy from original world structure)
    ubnd_a = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    lbnd_a = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    ubnd_a[:, :, 0] = 5.0  # spot upper bound
    ubnd_a[:, :, 1] = 5.0  # option upper bound
    lbnd_a[:, :, 0] = -5.0  # spot lower bound
    lbnd_a[:, :, 1] = -5.0  # option lower bound
    
    # Time features
    time_left_values = np.array(unique_times, dtype=val_world.np_dtype)
    time_left_features = np.tile(time_left_values[np.newaxis, :], (nSamples, 1))
    sqrt_time_left_features = np.sqrt(np.maximum(time_left_features, 0.0))
    
    # Prices for features
    prices = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    prices[:, :, 0] = spot_path[:, :nSteps]
    prices[:, :, 1] = option_prices
    
    # Update world dimensions
    val_world.nSamples = nSamples
    val_world.nSteps = nSteps
    val_world.nInst = nInst
    val_world.dt = time_left_values[0] - time_left_values[1] if len(time_left_values) > 1 else 1.0
    
    # Overwrite market data
    val_world.data.market.hedges = hedges
    val_world.data.market.cost = cost
    val_world.data.market.ubnd_a = ubnd_a
    val_world.data.market.lbnd_a = lbnd_a
    val_world.data.market.payoff = payoff
    
    # Overwrite features
    val_world.data.features.per_step.cost = cost
    val_world.data.features.per_step.price = prices
    val_world.data.features.per_step.ubnd_a = ubnd_a
    val_world.data.features.per_step.lbnd_a = lbnd_a
    val_world.data.features.per_step.time_left = time_left_features
    val_world.data.features.per_step.sqrt_time_left = sqrt_time_left_features
    val_world.data.features.per_step.spot = spot_path[:, :nSteps]
    val_world.data.features.per_step.ivol = ivols
    val_world.data.features.per_step.call_price = option_prices
    val_world.data.features.per_step.call_delta = option_deltas
    val_world.data.features.per_step.call_vega = option_vegas
    val_world.data.features.per_step.cost_v = cost[:, :, 1:2]
    
    # Update dummy dimension
    val_world.data.features.per_path[DIM_DUMMY] = (payoff * 0.)[:, np.newaxis]
    
    # Overwrite details
    val_world.details.spot_all = spot_path
    val_world.details.drift = np.full((nSamples, nSteps), 0.1, dtype=val_world.np_dtype)  # Placeholder
    val_world.details.rvol = ivols  # Use implied vol as proxy for realized vol
    
    # Regenerate TensorFlow data
    val_world.tf_data = tf_dict(
        features=val_world.data.features,
        market=val_world.data.market,
        dtype=val_world.tf_dtype
    )
    
    # Update sample weights
    val_world.sample_weights = np.full((nSamples, 1), 1.0/float(nSamples), dtype=val_world.np_dtype)
    val_world.tf_sample_weights = base.tf.constant(val_world.sample_weights, dtype=val_world.tf_dtype)
    val_world.sample_weights = val_world.sample_weights.reshape((nSamples,))
    val_world.tf_y = base.tf.zeros((nSamples,), dtype=val_world.tf_dtype)
    
    # Update timeline
    val_world.timeline = np.cumsum(np.linspace(0., nSteps, nSteps+1, 
                                             endpoint=True, dtype=np.float32)) * val_world.dt
    
    print("Successfully overwrote validation world with CSV data!")
    print(f"Final dimensions: {nSamples} samples, {nSteps} steps, {nInst} instruments")
    
    return val_world


def test_agent_with_csv(gym, world, csv_path, **kwargs):
    """
    Complete workflow: clone world, overwrite with CSV, test agent.
    
    Parameters
    ----------
    gym : VanillaDeepHedgingGym
        Trained gym
    world : SimpleWorld_Spot_ATM
        Original training world
    csv_path : str
        Path to CSV file
    **kwargs
        Additional arguments for overwrite_world_with_csv
    """
    print("="*60)
    print("TESTING AGENT WITH CSV DATA")
    print("="*60)
    
    # Clone validation world
    print("Cloning validation world...")
    val_world = world.clone(seed=999999, samples=100)  # Temporary size, will be overwritten
    
    # Overwrite with CSV data
    print(f"Loading CSV data from: {csv_path}")
    val_world = overwrite_world_with_csv(val_world, csv_path, **kwargs)
    
    # Test the agent
    print("Testing agent on CSV data...")
    results = gym(val_world.tf_data)
    
    # Extract results
    results_np = {key: val.numpy() if hasattr(val, 'numpy') else val 
                  for key, val in results.items()}
    
    # Calculate statistics
    weights = val_world.sample_weights
    stats = {
        'mean_utility': np.average(results_np['utility'], weights=weights),
        'mean_utility0': np.average(results_np['utility0'], weights=weights),
        'mean_loss': np.average(results_np['loss'], weights=weights),
        'mean_pnl': np.average(results_np['pnl'], weights=weights),
        'mean_cost': np.average(results_np['cost'], weights=weights),
        'mean_gains': np.average(results_np['gains'], weights=weights),
        'std_gains': np.sqrt(np.average((results_np['gains'] - 
                                        np.average(results_np['gains'], weights=weights))**2, 
                                       weights=weights)),
        'min_gains': np.min(results_np['gains']),
        'max_gains': np.max(results_np['gains'])
    }
    
    # Print results
    print("\nRESULTS:")
    print(f"Mean Utility: {stats['mean_utility']:.6f}")
    print(f"Mean Utility0: {stats['mean_utility0']:.6f}")
    print(f"Mean Loss: {stats['mean_loss']:.6f}")
    print(f"Mean PnL: {stats['mean_pnl']:.6f}")
    print(f"Mean Cost: {stats['mean_cost']:.6f}")
    print(f"Mean Gains: {stats['mean_gains']:.6f}")
    print(f"Std Gains: {stats['std_gains']:.6f}")
    print(f"Min/Max Gains: [{stats['min_gains']:.6f}, {stats['max_gains']:.6f}]")
    print("="*60)
    
    return results_np, val_world, stats


# Example usage
if __name__ == "__main__":
    # Create sample CSV data for testing
    sample_data = {
        'strike': [57, 57, 57, 58, 58, 58, 59, 59],
        'price': [57.69, 58.11, 57.32, 57.99, 58.98, 60.72, 58.03, 57.51],
        'ivol': [0.4114, 0.4055, 0.3785, 0.3337, 0.3239, 0.3203, 0.3316, 0.3301],
        'time_left': [21, 20, 19, 21, 20, 19, 21, 20],
        'spot': [57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_market_data.csv', index=False)
    print("Created sample CSV file: sample_market_data.csv")
    print(df)
    
    print("\nTo use this script:")
    print("1. Load your trained gym and world")
    print("2. Call: results, val_world, stats = test_agent_with_csv(gym, world, 'your_data.csv')")
    print("3. Or use overwrite_world_with_csv() directly for more control") 