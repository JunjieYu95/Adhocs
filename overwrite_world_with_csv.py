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
    """Load and validate CSV data with STRIPE-based structure."""
    df = pd.read_csv(csv_path)
    
    # Map column names to standard names (case insensitive)
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'stripe' in col_lower:
            column_mapping[col] = 'stripe'
        elif 'strike' in col_lower:
            column_mapping[col] = 'strike'
        elif 'price' in col_lower and 'strike' not in col_lower:
            column_mapping[col] = 'price'
        elif 'volatility' in col_lower or 'ivol' in col_lower:
            column_mapping[col] = 'ivol'
        elif 'time' in col_lower and 'maturity' in col_lower:
            column_mapping[col] = 'time_left'
        elif col_lower in ['time_left', 'time_to_maturity']:
            column_mapping[col] = 'time_left'
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Check required columns
    required_cols = ['stripe', 'strike', 'price', 'ivol', 'time_left']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        available_cols = list(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {available_cols}")
    
    # Keep only required columns
    df = df[required_cols].copy()
    
    # Sort by stripe and time_left (descending)
    df = df.sort_values(['stripe', 'time_left'], ascending=[True, False])
    
    # Add spot column
    df['spot'] = df['price']
    
    
    print(f"Loaded CSV data: {len(df)} rows")
    print(f"Number of stripes (paths): {df['stripe'].nunique()}")
    print(f"Time steps per stripe:")
    for stripe in df['stripe'].unique()[:5]:  # Show first 5 stripes
        stripe_data = df[df['stripe'] == stripe]
        print(f"  {stripe}: {len(stripe_data)} steps, strike={stripe_data['strike'].iloc[0]}, time range={stripe_data['time_left'].max()}-{stripe_data['time_left'].min()}")
    
    # Check that each stripe has the same strike throughout
    strike_consistency = df.groupby('stripe')['strike'].nunique()
    inconsistent_stripes = strike_consistency[strike_consistency > 1]
    if len(inconsistent_stripes) > 0:
        print(f"Warning: {len(inconsistent_stripes)} stripes have inconsistent strikes")
    
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
    
    # Get unique stripes (each stripe is a sample/path)
    unique_stripes = df['stripe'].unique()
    nSamples = len(unique_stripes)
    
    # Get time steps from first stripe
    first_stripe_data = df[df['stripe'] == unique_stripes[0]].sort_values('time_left', ascending=False)
    unique_times = first_stripe_data['time_left'].values
    nSteps = len(unique_times)
    nInst = 1  # only spot
    
    print(f"New dimensions: nSamples={nSamples}, nSteps={nSteps}, nInst={nInst}")
    print(f"Time steps: {unique_times}")
    
    # Initialize arrays
    spot_path = np.zeros((nSamples, nSteps + 1), dtype=val_world.np_dtype)
    ivols = np.zeros((nSamples, nSteps), dtype=val_world.np_dtype)
    time_left = np.zeros((nSamples, nSteps), dtype=val_world.np_dtype)
    spot_path[:, 0] = df['spot'].values

    
    # Fill arrays with CSV data - each stripe becomes one sample
    for sample_idx, stripe in enumerate(unique_stripes):
        stripe_data = df[df['stripe'] == stripe].sort_values('time_left', ascending=False)
        
        # Ensure we have the expected number of time steps
        if len(stripe_data) != nSteps:
            print(f"Warning: Stripe {stripe} has {len(stripe_data)} time steps, expected {nSteps}")
            # Pad or truncate as needed
            if len(stripe_data) < nSteps:
                # Pad with last row
                last_row = stripe_data.iloc[-1:].copy()
                while len(stripe_data) < nSteps:
                    stripe_data = pd.concat([stripe_data, last_row], ignore_index=True)
            else:
                # Truncate
                stripe_data = stripe_data.head(nSteps)
        
        # Fill arrays for this sample
        strikes[sample_idx] = stripe_data['strike'].iloc[0]  # Strike should be same throughout
        spot_path[sample_idx, 0] = stripe_data['spot'].iloc[0]  # Initial spot
        
        # Fill time series data
        for t_idx in range(nSteps):
            if t_idx < len(stripe_data):
                ivols[sample_idx, t_idx] = stripe_data['ivol'].iloc[t_idx]
                time_left[sample_idx, t_idx] = stripe_data['time_left'].iloc[t_idx]
            else:
                # Use last available values if data is shorter
                ivols[sample_idx, t_idx] = ivols[sample_idx, t_idx-1]
                time_left[sample_idx, t_idx] = time_left[sample_idx, t_idx-1]
    
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
    
    
    # Combine hedges
    hedges = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    hedges[:, :, 0] = spot_returns  # Spot
    
    # Calculate costs
    cost_spot = spot_path[:, :nSteps] * cost_s
    
    cost = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    cost[:, :, 0] = cost_spot
    
    # Action bounds (copy from original world structure)
    ubnd_a = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    lbnd_a = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    ubnd_a[:, :, 0] = 1.0  # spot upper bound
    ubnd_a[:, :, 1] = 1.0  # option upper bound
    lbnd_a[:, :, 0] = -1.0  # spot lower bound
    lbnd_a[:, :, 1] = -1.0  # option lower bound
    
    # Time features
    time_left_values = np.array(unique_times, dtype=val_world.np_dtype)
    time_left_features = np.tile(time_left_values[np.newaxis, :], (nSamples, 1))
    sqrt_time_left_features = np.sqrt(np.maximum(time_left_features, 0.0))
    
    # Prices for features
    prices = np.zeros((nSamples, nSteps, nInst), dtype=val_world.np_dtype)
    prices[:, :, 0] = spot_path[:, :nSteps]
    
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
    # Create sample CSV data for testing in the STRIPE format
    sample_data = []
    
    # Create 3 stripes (paths) with 5 time steps each
    stripes = ['2015-06-01', '2015-06-02', '2015-06-03']
    strikes = [57.0, 58.0, 59.0]
    
    for i, (stripe, strike) in enumerate(zip(stripes, strikes)):
        # Create time series for this stripe (path)
        for j, time_left in enumerate([21, 20, 19, 18, 17]):
            price = 3.0 - j * 0.4 + i * 0.2  # Decreasing price over time, different levels per stripe
            ivol = 0.40 - j * 0.02 + i * 0.01  # Decreasing vol over time
            
            sample_data.append({
                'stripe': stripe,
                'strike': strike,
                'price': price,
                'ivol': ivol,
                'time_left': time_left
            })
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_market_data.csv', index=False)
    print("Created sample CSV file: sample_market_data.csv")
    print("Sample data structure (STRIPE format):")
    print(df.head(10))
    
    print(f"\nData summary:")
    print(f"- {len(stripes)} stripes (paths)")
    print(f"- {len(df) // len(stripes)} time steps per stripe")
    print(f"- Strikes: {strikes}")
    
    print("\nTo use this script with your data:")
    print("1. Ensure your CSV has columns: STRIPE, STRIKE, Price, VOLATILITY, Time_to_Maturity")
    print("2. Each STRIPE represents one path/sample")
    print("3. Each path should have the same STRIKE throughout")
    print("4. Call: results, val_world, stats = test_agent_with_csv(gym, world, 'your_data.csv')")
    print("5. Or use overwrite_world_with_csv() directly for more control") 



## 