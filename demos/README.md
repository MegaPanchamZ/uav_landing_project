# Demos

This directory contains demonstration scripts that showcase the UAV Landing System capabilities.

## Scripts

- **`uav_demo_end_to_end.py`** - Comprehensive end-to-end demo using UDD6 dataset
  - Demonstrates both neural and neuro-symbolic analysis
  - Generates visualization comparing results
  - Uses real drone imagery for testing

- **`demo_complete_system.py`** - Complete system demo with Scallop integration
  - Shows enhanced UAV detector with symbolic reasoning
  - Creates synthetic test scenarios
  - Demonstrates neuro-symbolic capabilities

- **`demo_summary_generator.py`** - Generates comprehensive demo reports
  - Creates detailed performance summaries
  - Exports results to JSON format
  - Provides formatted status reports

## Output Files

- **`demo_summary.json`** - JSON summary of demo results
- **`uav_demo_results.png`** - Visualization of demo results (10.7MB)

## Usage

From the project root directory:

```bash
# Run end-to-end demo with UDD6 dataset
python demos/uav_demo_end_to_end.py

# Run complete system demo
python demos/demo_complete_system.py

# Generate demo summary
python demos/demo_summary_generator.py
```

All demos are configured to work with the organized project structure and include proper path handling.
