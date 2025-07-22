# Testing Deep Hedging Agents on Real Market Data

This guide explains how to test your trained deep hedging agents on realistic market data using the `RealDataAdapter` class.

## Overview

The main challenge in testing deep hedging agents on real data is that the framework expects data in a specific format (hedge returns, features, etc.), while real market data typically contains option prices, strikes, implied volatilities, etc. The `RealDataAdapter` class solves this by transforming real market data into the format expected by the gym.

## Quick Start

```python
from deephedging.real_data_adapter import RealDataAdapter, test_trained_agent
from deephedging.gym import VanillaDeepHedgingGym
import pandas as pd

# 1. Load your market data
market_data = pd.read_csv("your_market_data.csv")

# 2. Create adapter 
real_world = RealDataAdapter(market_data)

# 3. Load your trained gym (replace with your loading logic)
gym = VanillaDeepHedgingGym.load_from_cache("path/to/your/model")

# 4. Test the agent
results = test_trained_agent(gym, real_world)
```

## Required Data Format

Your market data should contain the following columns:

### Required Columns
- `strike`: Strike prices of the options
- `price`: Option prices (mid-prices preferred)
- `ivol`: Implied volatilities (as decimals, e.g., 0.20 for 20%)
- `time_left`: Time to expiration (in days, or consistent units)

### Optional Columns
- `spot`: Underlying spot price (will use first strike as proxy if missing)
- `delta`: Option delta (will be calculated using Black-Scholes if missing)
- `vega`: Option vega (will be calculated using Black-Scholes if missing)
- `timestamp`: Time stamps for the data

### Example Data Structure

```python
data = {
    'strike': [55, 56, 57, 58, 59, 60],
    'price': [2.45, 1.80, 1.25, 0.85, 0.55, 0.35],
    'ivol': [0.22, 0.21, 0.20, 0.21, 0.22, 0.23],
    'time_left': [21, 21, 21, 21, 21, 21],
    'spot': [57.0, 57.0, 57.0, 57.0, 57.0, 57.0]
}
df = pd.DataFrame(data)
```

## Configuration Options

You can customize the adapter behavior using a `Config` object:

```python
from deephedging.base import Config

config = Config()

# Market parameters
config.cost_s = 0.0002        # Spot trading cost (default: 0.0002)
config.cost_v = 0.02          # Vega trading cost (default: 0.02)
config.cost_p = 0.0005        # Additional option cost (default: 0.0005)

# Action bounds
config.ubnd_as = 5.0          # Upper bound for spot actions
config.lbnd_as = -5.0         # Lower bound for spot actions
config.ubnd_av = 5.0          # Upper bound for option actions
config.lbnd_av = -5.0         # Lower bound for option actions

# Data processing
config.max_time_steps = 21    # Maximum number of time steps
config.interpolate_missing = True  # Interpolate missing data points

# Payoff definition
config.payoff_type = "short_call"  # "short_call", "short_put", or "custom"

# Create adapter with custom config
real_world = RealDataAdapter(market_data, config)
```

## Loading Data from Different Sources

### From CSV File
```python
real_world = RealDataAdapter.from_csv("market_data.csv")
```

### From Excel File
```python
real_world = RealDataAdapter.from_excel("market_data.xlsx", sheet_name="Options")
```

### From Your Data Source
```python
# Load from your database, API, etc.
data = load_from_your_source()
market_data = pd.DataFrame(data)
real_world = RealDataAdapter(market_data)
```

## Understanding the Output

The `test_trained_agent()` function returns detailed results:

```python
results = test_trained_agent(gym, real_world, verbose=True)

# Key metrics
print(f"Mean Utility: {results['mean_utility']:.6f}")
print(f"Mean PnL: {results['mean_pnl']:.6f}")
print(f"Mean Trading Cost: {results['mean_cost']:.6f}")
print(f"Mean Total Gains: {results['mean_gains']:.6f}")
print(f"Std of Gains: {results['std_gains']:.6f}")

# Full results for each sample
full_results = results['full_results']
actions = full_results['actions']  # [nSamples, nSteps, nInstruments]
pnl = full_results['pnl']         # [nSamples]
cost = full_results['cost']       # [nSamples]
```

## Data Transformation Details

The adapter performs several key transformations:

1. **Hedge Returns Calculation**: Converts option prices to hedge returns (dH_t = H_T - H_t)
2. **Missing Data Handling**: Interpolates or estimates missing Greeks using Black-Scholes
3. **Time Step Organization**: Groups data by time-to-expiration and creates time steps
4. **Feature Engineering**: Creates all required features (time_left, sqrt_time_left, etc.)
5. **Cost Modeling**: Estimates trading costs based on delta, vega, and bid-ask spreads

### Hedge Returns Formula

For an option with price H_t at time t and payoff H_T at maturity:
```
Hedge Return = H_T - H_t
```

Where:
- H_t = Current option price from your data
- H_T = Option payoff at maturity (calculated based on final spot and strike)

## Example: Complete Workflow

```python
#!/usr/bin/env python3
import pandas as pd
from deephedging.real_data_adapter import RealDataAdapter, test_trained_agent
from deephedging.base import Config

def test_my_agent():
    # 1. Load your market data
    market_data = pd.read_csv("my_option_data.csv")
    
    # 2. Configure the adapter
    config = Config()
    config.payoff_type = "short_call"  # We're short call options
    config.cost_s = 0.0003  # Higher spot trading cost
    config.max_time_steps = 30  # More time steps
    
    # 3. Create the adapter
    real_world = RealDataAdapter(market_data, config)
    
    # 4. Load your trained agent (replace with your method)
    gym = load_my_trained_agent()  # Your function here
    
    # 5. Test the agent
    results = test_trained_agent(gym, real_world, verbose=True)
    
    # 6. Analyze results
    if results['mean_utility'] > -0.05:
        print("✓ Agent performs well on real data!")
    else:
        print("⚠ Consider investigating performance")
    
    return results

if __name__ == "__main__":
    results = test_my_agent()
```

## Troubleshooting

### Common Issues and Solutions

1. **"Missing required columns" Error**
   - Ensure your data has: `strike`, `price`, `ivol`, `time_left`
   - Check column names match exactly (case-sensitive)

2. **Poor Performance on Real Data**
   - Verify data quality (no negative prices, reasonable volatilities)
   - Check if features used by agent are available in real data
   - Consider retraining with more diverse synthetic data

3. **Shape Mismatch Errors**
   - Ensure consistent number of strikes across time steps
   - Check that time_left values are reasonable
   - Verify no missing data in critical columns

4. **Unrealistic Trading Costs**
   - Adjust cost_s, cost_v, cost_p parameters
   - Consider market impact for large positions
   - Account for bid-ask spreads in your data

### Data Quality Checks

Before testing, verify your data:

```python
# Check data quality
print(f"Data shape: {market_data.shape}")
print(f"Time range: {market_data['time_left'].min()} to {market_data['time_left'].max()}")
print(f"Strike range: {market_data['strike'].min()} to {market_data['strike'].max()}")
print(f"Price range: {market_data['price'].min()} to {market_data['price'].max()}")
print(f"IV range: {market_data['ivol'].min():.1%} to {market_data['ivol'].max():.1%}")

# Check for missing data
print(f"Missing values:\n{market_data.isnull().sum()}")
```

## Advanced Usage

### Custom Payoff Functions

```python
def my_custom_payoff(spots, strikes):
    """Custom payoff function"""
    # Example: Barrier option
    barrier = 60.0
    breached = np.any(spots > barrier, axis=1)
    regular_payoff = -np.maximum(spots[:, -1] - strikes, 0.0)
    return np.where(breached, 0.0, regular_payoff)

config = Config()
config.payoff_func = my_custom_payoff
config.payoff_type = "custom"
```

### Multiple Time Series

```python
# Test on multiple datasets
datasets = ["data1.csv", "data2.csv", "data3.csv"]
results = []

for dataset in datasets:
    market_data = pd.read_csv(dataset)
    real_world = RealDataAdapter(market_data)
    result = test_trained_agent(gym, real_world, verbose=False)
    results.append(result)
    
# Compare performance across datasets
for i, result in enumerate(results):
    print(f"Dataset {i+1}: Utility = {result['mean_utility']:.6f}")
```

## Integration with Existing Code

The `RealDataAdapter` creates objects compatible with the existing deep hedging framework:

```python
# Works with existing plotting functions
from deephedging.plot_training import Plotter
plotter = Plotter(real_world, real_world.clone())

# Compatible with existing analysis tools  
summary_stats = real_world.get_market_summary()

# Can be used in place of synthetic worlds
comparison_results = gym(real_world.tf_data)
```

## Performance Considerations

- **Memory Usage**: Large datasets may require chunking or sampling
- **Computation Time**: Black-Scholes calculations for missing Greeks can be slow
- **Data Preprocessing**: Consider preprocessing data offline for repeated testing

For large-scale testing, consider:

```python
# Sample subset for initial testing
sampled_data = market_data.sample(n=1000)
real_world = RealDataAdapter(sampled_data)

# Or process in chunks
chunk_size = 5000
for chunk in pd.read_csv("large_data.csv", chunksize=chunk_size):
    real_world = RealDataAdapter(chunk)
    results = test_trained_agent(gym, real_world)
    # Process results...
```

## Next Steps

1. **Test on Historical Data**: Use historical option chains to backtest performance
2. **Live Testing**: Integrate with real-time data feeds for live testing
3. **Performance Attribution**: Analyze which market conditions drive performance
4. **Model Validation**: Compare results with traditional hedging strategies

For more details, see the example script `example_test_real_data.py`. 