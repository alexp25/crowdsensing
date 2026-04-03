py -3.12 -m pip install -r requirements.txt

using Mesa
https://mesa.readthedocs.io/stable/

household_model.py
- agent-based simulation of household demand groups interacting with constrained water service zones
- supports multiple behavior profiles and multiple daily activation windows per profile
- tracks demand load, queue pressure, pressure index/deficit, and estimated consumption (L/min)

script_household_water.py
- Scenario 1: baseline with three separated groups
  (families with children, working-adult households, workplaces/schools)
- Scenario 2: targeted interventions on the same three groups
- prints summary metrics and multi-seed comparisons
- saves plots to output/household_water_scenarios.png
