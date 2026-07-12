"""
Registry of signal pairs to analyze.

Add a new pair by:
  1. (If needed) Adding a CSV to data/ and a loader function to sources.py
  2. Defining the two Signal objects + the SignalPair below
  3. Adding to the PAIRS dict at the bottom

To run all pairs:
    python run.py --all

To run one pair:
    python run.py copper_transformer
"""

from core import Signal, SignalPair
from sources import load_fred_monthly, load_annual


# =================================================================
# SIGNAL DEFINITIONS
# =================================================================

copper_ppi = Signal(
    name="copper_ppi",
    description="PPI by Commodity: Special Indexes: Copper and Copper Products",
    source_url="https://fred.stlouisfed.org/series/WPUSI019011",
    units="Index 1982=100, Not Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("copper_ppi",),
)

transformer_ppi = Signal(
    name="transformer_ppi",
    description="PPI by Industry: Electric Power and Specialty Transformer Manufacturing",
    source_url="https://fred.stlouisfed.org/series/PCU335311335311",
    units="Index Jun 1981=100, Not Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("transformer_ppi",),
)

hyperscaler_capex = Signal(
    name="hyperscaler_capex",
    description="Combined annual capex for Amazon + Google + Meta + Microsoft",
    source_url="https://platformonomics.com/2026/02/follow-the-capex-2025-retrospective/",
    units="USD billions, calendar year",
    loader=load_annual,
    loader_args=("hyperscaler_capex.csv",),
)

aluminum_ppi = Signal(
    name="aluminum_ppi",
    description="PPI by Commodity: Metals and Metal Products: Aluminum Mill Shapes",
    source_url="https://fred.stlouisfed.org/series/WPU102501",
    units="Index 1982=100, Not Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("aluminum_ppi",),
)

steel_ppi = Signal(
    name="steel_ppi",
    description="PPI by Commodity: Metals and Metal Products: Iron and Steel",
    source_url="https://fred.stlouisfed.org/series/WPU101",
    units="Index 1982=100, Not Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("steel_ppi",),
)

manufacturing_employment = Signal(
    name="manufacturing_employment",
    description="All Employees, Manufacturing (BLS Current Employment Statistics)",
    source_url="https://fred.stlouisfed.org/series/MANEMP",
    units="Thousands of Persons, Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("manufacturing_employment",),
)

copper_global_price = Signal(
    name="copper_global_price",
    description="Global price of Copper (IMF Primary Commodity Prices)",
    source_url="https://fred.stlouisfed.org/series/PCOPPUSDM",
    units="U.S. Dollars per Metric Ton, Not Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("copper_global_price",),
)

wti_oil = Signal(
    name="wti_oil",
    description="Spot Crude Oil Price: West Texas Intermediate (WTI)",
    source_url="https://fred.stlouisfed.org/series/WTISPLC",
    units="Dollars per Barrel, Not Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("wti_oil",),
)

nickel_global_price = Signal(
    name="nickel_global_price",
    description="Global price of Nickel (IMF Primary Commodity Prices)",
    source_url="https://fred.stlouisfed.org/series/PNICKUSDM",
    units="U.S. Dollars per Metric Ton, Not Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("nickel_global_price",),
)

steel_scrap_ppi = Signal(
    name="steel_scrap_ppi",
    description="PPI by Commodity: Metals and Metal Products: Iron and Steel Scrap",
    source_url="https://fred.stlouisfed.org/series/WPU1012",
    units="Index 1982=100, Not Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("steel_scrap_ppi",),
)

electrical_equip_ppi = Signal(
    name="electrical_equip_ppi",
    description="PPI by Commodity: Machinery and Equipment: Electrical Machinery and Equipment",
    source_url="https://fred.stlouisfed.org/series/WPU117",
    units="Index 1982=100, Not Seasonally Adjusted",
    loader=load_fred_monthly,
    loader_args=("electrical_equip_ppi",),
)


# =================================================================
# SIGNAL PAIR DEFINITIONS
# =================================================================

PAIRS = {
    "copper_transformer": SignalPair(
        name="copper_transformer",
        leading=copper_ppi,
        outcome=transformer_ppi,
        hypothesis="Copper price changes lead transformer PPI by 6-12 months as copper is the primary metal input to transformer windings",
        expected_lead_low=6,
        expected_lead_high=12,
        frequency="monthly",
    ),

    "hyperscaler_transformer": SignalPair(
        name="hyperscaler_transformer",
        leading=hyperscaler_capex,
        outcome=transformer_ppi,
        hypothesis="Hyperscaler capex (data center buildout) leads transformer PPI by 1-2 years as AI demand drives grid equipment demand",
        expected_lead_low=1,
        expected_lead_high=2,
        frequency="annual",
    ),

    "aluminum_transformer": SignalPair(
        name="aluminum_transformer",
        leading=aluminum_ppi,
        outcome=transformer_ppi,
        hypothesis="Aluminum prices lead transformer PPI by 6-12 months — aluminum is used in transformer windings and housings alongside copper",
        expected_lead_low=6,
        expected_lead_high=12,
        frequency="monthly",
    ),

    "steel_transformer": SignalPair(
        name="steel_transformer",
        leading=steel_ppi,
        outcome=transformer_ppi,
        hypothesis="Iron and steel prices lead transformer PPI by 6-12 months — steel (especially grain-oriented electrical steel) is the core material of transformer cores",
        expected_lead_low=6,
        expected_lead_high=12,
        frequency="monthly",
    ),

    "manufacturing_employment_transformer": SignalPair(
        name="manufacturing_employment_transformer",
        leading=manufacturing_employment,
        outcome=transformer_ppi,
        hypothesis="US manufacturing employment leads transformer PPI by 3-9 months — labor market reflects industrial activity which drives equipment demand",
        expected_lead_low=3,
        expected_lead_high=9,
        frequency="monthly",
    ),

    "copper_global_transformer": SignalPair(
        name="copper_global_transformer",
        leading=copper_global_price,
        outcome=transformer_ppi,
        hypothesis="Global copper price (IMF) leads transformer PPI by 4-10 months — copper is the primary winding metal; an independent copper measure to cross-check the copper PPI signal",
        expected_lead_low=4,
        expected_lead_high=10,
        frequency="monthly",
    ),

    "oil_transformer": SignalPair(
        name="oil_transformer",
        leading=wti_oil,
        outcome=transformer_ppi,
        hypothesis="Crude oil (WTI) leads transformer PPI by 3-12 months — energy is an input to manufacturing and transport, so oil cost may feed into equipment prices",
        expected_lead_low=3,
        expected_lead_high=12,
        frequency="monthly",
    ),

    "nickel_transformer": SignalPair(
        name="nickel_transformer",
        leading=nickel_global_price,
        outcome=transformer_ppi,
        hypothesis="Global nickel price leads transformer PPI by 4-10 months — nickel is an alloying input to electrical steels and stainless components",
        expected_lead_low=4,
        expected_lead_high=10,
        frequency="monthly",
    ),

    "steelscrap_transformer": SignalPair(
        name="steelscrap_transformer",
        leading=steel_scrap_ppi,
        outcome=transformer_ppi,
        hypothesis="Iron and steel scrap price leads transformer PPI by 4-12 months — scrap is the raw feed for electric-arc steelmaking, upstream of the finished steel used in transformer cores",
        expected_lead_low=4,
        expected_lead_high=12,
        frequency="monthly",
    ),

    "electricalequip_transformer": SignalPair(
        name="electricalequip_transformer",
        leading=electrical_equip_ppi,
        outcome=transformer_ppi,
        hypothesis="Broad electrical machinery and equipment PPI leads transformer PPI by 0-6 months — transformers are a subset of electrical equipment, so the broader category may move slightly ahead (note: likely a sibling/contemporaneous series, not a true upstream driver)",
        expected_lead_low=0,
        expected_lead_high=6,
        frequency="monthly",
    ),
}
