"""
Energy Yield Calculator - Malaysian Solar Context
==================================================
Estimates annual energy production using Malaysian irradiance data,
temperature derating, system losses, and ATAP/NEM financial analysis.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from config import (
    PANEL_SPEC, LAYOUT, PERFORMANCE,
    MALAYSIA_IRRADIANCE, TARIFF
)


class EnergyCalculator:
    """
    Calculates expected energy yield and financial returns for a
    rooftop PV system in Malaysia.
    """

    def __init__(self, region: str = "Default"):
        """
        Args:
            region: Malaysian region key from MALAYSIA_IRRADIANCE config
        """
        self.set_region(region)
        self.system_kwp = 0.0
        self.n_panels = 0
        self.tilt_deg = LAYOUT["tilt_angle_deg"]
        self.azimuth_deg = LAYOUT["azimuth_deg"]

    def set_region(self, region: str):
        """Set the irradiance region."""
        if region not in MALAYSIA_IRRADIANCE:
            print(f"[WARNING] Region '{region}' not found. Using 'Default'.")
            region = "Default"
        self.region = region
        irr = MALAYSIA_IRRADIANCE[region]
        self.daily_ghi = irr[0]       # kWh/m²/day
        self.peak_sun_hours = irr[1]   # hours
        self.avg_temp_c = irr[2]       # °C

    def set_system(self, n_panels: int, tilt_deg: float = None, azimuth_deg: float = None):
        """Configure the PV system parameters."""
        self.n_panels = n_panels
        self.system_kwp = n_panels * PANEL_SPEC["rated_power_wp"] / 1000.0
        if tilt_deg is not None:
            self.tilt_deg = tilt_deg
        if azimuth_deg is not None:
            self.azimuth_deg = azimuth_deg

    # ============================================================
    # IRRADIANCE ON TILTED SURFACE
    # ============================================================
    def calculate_poa_irradiance(self) -> float:
        """
        Calculate Plane of Array (POA) irradiance for tilted surface.
        Uses simplified isotropic sky model suitable for low-tilt Malaysia.

        Returns:
            Annual POA irradiance in kWh/m²
        """
        # For Malaysia (near equator), optimal tilt is ~0-15°
        # Tilt factor approximation for low latitudes
        latitude_rad = np.radians(3.0)  # ~3°N average for Malaysia
        tilt_rad = np.radians(self.tilt_deg)

        # Simplified transposition factor
        # For near-equatorial locations with low tilt, factor ≈ 1.0-1.05
        if self.tilt_deg <= 15:
            tilt_factor = 1.0 + 0.003 * self.tilt_deg  # Small boost for tilt
        else:
            tilt_factor = 1.0 + 0.05 * np.cos(tilt_rad - latitude_rad)

        # Azimuth correction (ideal is south-facing = 180°)
        azimuth_deviation = abs(self.azimuth_deg - 180)
        azimuth_factor = 1.0 - 0.002 * min(azimuth_deviation, 90)

        daily_poa = self.daily_ghi * tilt_factor * azimuth_factor
        annual_poa = daily_poa * 365  # kWh/m²/year

        return annual_poa

    # ============================================================
    # TEMPERATURE DERATING
    # ============================================================
    def calculate_temperature_loss(self) -> float:
        """
        Calculate power loss due to cell temperature.
        Uses NOCT method for Malaysian ambient conditions.

        Returns:
            Temperature derating factor (0 to 1)
        """
        # Cell temperature estimate using NOCT
        noct = PANEL_SPEC["noct"]
        irradiance_w_m2 = self.daily_ghi * 1000 / self.peak_sun_hours

        # Cell temperature
        t_cell = self.avg_temp_c + (noct - 20) * (irradiance_w_m2 / 800)

        # Temperature derating from STC (25°C)
        temp_diff = t_cell - 25.0  # STC reference
        temp_coeff = PANEL_SPEC["temp_coefficient"]  # e.g., -0.0035
        derating_factor = 1.0 + temp_coeff * temp_diff

        return max(0.5, min(1.0, derating_factor))  # Clamp

    # ============================================================
    # ANNUAL ENERGY YIELD
    # ============================================================
    def calculate_annual_yield(self) -> Dict:
        """
        Calculate detailed annual energy yield.

        Returns:
            Dict with yield breakdown and all contributing factors
        """
        if self.system_kwp <= 0:
            return {"error": "System not configured. Call set_system() first."}

        # Step 1: POA Irradiance
        annual_poa = self.calculate_poa_irradiance()  # kWh/m²/year

        # Step 2: Gross DC energy (before losses)
        # E_dc = P_stc × (H_poa / G_stc) × N_panels
        # where G_stc = 1000 W/m² (Standard Test Conditions irradiance)
        gross_dc_kwh = self.system_kwp * (annual_poa / 1000.0) * 1000  # kWh

        # Step 3: Temperature derating
        temp_factor = self.calculate_temperature_loss()

        # Step 4: System losses
        perf = PERFORMANCE
        soiling = 1 - perf["soiling_loss"]
        mismatch = 1 - perf["mismatch_loss"]
        wiring = 1 - perf["wiring_loss"]
        inverter_eff = perf["inverter_efficiency"]
        availability = perf["availability"]

        # Combined loss factor
        system_factor = (
            temp_factor * soiling * mismatch *
            wiring * inverter_eff * availability
        )

        # Step 5: Net annual yield
        net_annual_kwh = gross_dc_kwh * system_factor

        # Performance ratio
        performance_ratio = system_factor

        # Specific yield (kWh per kWp installed)
        specific_yield = net_annual_kwh / self.system_kwp if self.system_kwp > 0 else 0

        # Monthly estimate (simplified - Malaysia has relatively uniform insolation)
        monthly_kwh = self._estimate_monthly_yield(net_annual_kwh)

        return {
            "system_capacity_kwp": round(self.system_kwp, 2),
            "n_panels": self.n_panels,
            "region": self.region,
            "annual_poa_irradiance_kwh_m2": round(annual_poa, 1),
            "daily_ghi_kwh_m2": self.daily_ghi,
            "peak_sun_hours": self.peak_sun_hours,
            "avg_ambient_temp_c": self.avg_temp_c,
            "tilt_angle_deg": self.tilt_deg,
            "azimuth_deg": self.azimuth_deg,
            "gross_dc_energy_kwh": round(gross_dc_kwh, 0),
            "temperature_derating": round(temp_factor, 4),
            "soiling_factor": soiling,
            "mismatch_factor": mismatch,
            "wiring_factor": wiring,
            "inverter_efficiency": inverter_eff,
            "availability": availability,
            "combined_system_factor": round(system_factor, 4),
            "performance_ratio": round(performance_ratio, 4),
            "net_annual_yield_kwh": round(net_annual_kwh, 0),
            "specific_yield_kwh_kwp": round(specific_yield, 0),
            "monthly_yield_kwh": monthly_kwh,
            "daily_avg_yield_kwh": round(net_annual_kwh / 365, 1),
        }

    def _estimate_monthly_yield(self, annual_kwh: float) -> Dict[str, float]:
        """
        Estimate monthly energy yield.
        Malaysia has relatively uniform solar irradiance year-round,
        with slight variations due to monsoon seasons.
        """
        # Monthly irradiance variation factors for Malaysia
        # Northeast monsoon (Nov-Mar): slightly lower
        # Southwest monsoon (May-Sep): moderate
        # Inter-monsoon (Apr, Oct): highest
        monthly_factors = {
            "Jan": 0.078, "Feb": 0.080, "Mar": 0.088,
            "Apr": 0.090, "May": 0.086, "Jun": 0.082,
            "Jul": 0.083, "Aug": 0.085, "Sep": 0.084,
            "Oct": 0.088, "Nov": 0.080, "Dec": 0.076,
        }

        return {
            month: round(annual_kwh * factor, 0)
            for month, factor in monthly_factors.items()
        }

    # ============================================================
    # FINANCIAL ANALYSIS (ATAP / NEM)
    # ============================================================
    def calculate_financial_returns(
        self,
        annual_yield_kwh: Optional[float] = None,
        system_cost_override: Optional[float] = None
    ) -> Dict:
        """
        Calculate financial returns under Malaysian ATAP/NEM scheme.

        Args:
            annual_yield_kwh: Override annual yield (auto-calculates if None)
            system_cost_override: Override total system cost in MYR

        Returns:
            Dict with financial analysis results
        """
        if annual_yield_kwh is None:
            yield_data = self.calculate_annual_yield()
            annual_yield_kwh = yield_data["net_annual_yield_kwh"]

        # System cost
        if system_cost_override:
            total_cost = system_cost_override
        else:
            total_cost = self.system_kwp * 1000 * TARIFF["system_cost_myr_per_wp"]

        # Annual revenue (NEM savings)
        annual_savings = annual_yield_kwh * TARIFF["nem_rate_myr_kwh"]

        # Annual O&M cost
        annual_om = TARIFF["annual_maintenance_myr"]

        # Net annual benefit
        net_annual = annual_savings - annual_om

        # Simple payback period
        payback_years = total_cost / net_annual if net_annual > 0 else float('inf')

        # 25-year lifetime analysis with degradation
        lifetime_data = self._lifetime_analysis(
            annual_yield_kwh, total_cost, annual_om
        )

        # LCOE (Levelised Cost of Energy)
        total_lifetime_energy = lifetime_data["total_energy_kwh"]
        total_lifetime_cost = total_cost + (annual_om * TARIFF["system_lifetime_years"])
        lcoe = total_lifetime_cost / total_lifetime_energy if total_lifetime_energy > 0 else 0

        return {
            "total_system_cost_myr": round(total_cost, 0),
            "cost_per_wp_myr": TARIFF["system_cost_myr_per_wp"],
            "nem_rate_myr_kwh": TARIFF["nem_rate_myr_kwh"],
            "year1_energy_kwh": round(annual_yield_kwh, 0),
            "year1_savings_myr": round(annual_savings, 0),
            "annual_om_cost_myr": annual_om,
            "net_annual_benefit_myr": round(net_annual, 0),
            "simple_payback_years": round(payback_years, 1),
            "lcoe_myr_kwh": round(lcoe, 3),
            "lifetime_years": TARIFF["system_lifetime_years"],
            "total_lifetime_energy_kwh": round(lifetime_data["total_energy_kwh"], 0),
            "total_lifetime_savings_myr": round(lifetime_data["total_savings"], 0),
            "total_lifetime_profit_myr": round(lifetime_data["total_profit"], 0),
            "roi_pct": round(lifetime_data["roi_pct"], 1),
            "yearly_breakdown": lifetime_data["yearly"],
        }

    def _lifetime_analysis(
        self,
        year1_kwh: float,
        total_cost: float,
        annual_om: float
    ) -> Dict:
        """Calculate 25-year lifetime energy and financial projections."""
        perf = PERFORMANCE
        lifetime = TARIFF["system_lifetime_years"]
        nem_rate = TARIFF["nem_rate_myr_kwh"]

        yearly = []
        cumulative_energy = 0
        cumulative_savings = 0
        cumulative_cost = total_cost

        for year in range(1, lifetime + 1):
            # Apply degradation
            if year == 1:
                degradation = 1 - perf["degradation_year1"]
            else:
                degradation = (
                    (1 - perf["degradation_year1"]) *
                    (1 - perf["degradation_annual"]) ** (year - 1)
                )

            year_kwh = year1_kwh * degradation
            year_savings = year_kwh * nem_rate
            year_profit = year_savings - annual_om
            cumulative_energy += year_kwh
            cumulative_savings += year_savings
            cumulative_cost -= year_profit

            yearly.append({
                "year": year,
                "energy_kwh": round(year_kwh, 0),
                "savings_myr": round(year_savings, 0),
                "cumulative_profit_myr": round(
                    cumulative_savings - annual_om * year - total_cost, 0
                ),
            })

        total_savings = cumulative_savings
        total_profit = total_savings - (annual_om * lifetime) - total_cost
        roi_pct = (total_profit / total_cost * 100) if total_cost > 0 else 0

        return {
            "total_energy_kwh": cumulative_energy,
            "total_savings": total_savings,
            "total_profit": total_profit,
            "roi_pct": roi_pct,
            "yearly": yearly,
        }

    # ============================================================
    # FULL SYSTEM REPORT
    # ============================================================
    def generate_full_report(self) -> Dict:
        """Generate a complete system performance and financial report."""
        yield_data = self.calculate_annual_yield()
        financial = self.calculate_financial_returns(
            yield_data["net_annual_yield_kwh"]
        )

        return {
            "energy_yield": yield_data,
            "financial_analysis": financial,
            "system_summary": {
                "capacity_kwp": self.system_kwp,
                "panels": self.n_panels,
                "panel_power_wp": PANEL_SPEC["rated_power_wp"],
                "region": self.region,
                "tilt_deg": self.tilt_deg,
                "azimuth_deg": self.azimuth_deg,
                "performance_ratio": yield_data["performance_ratio"],
                "specific_yield": yield_data["specific_yield_kwh_kwp"],
            }
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================
def quick_estimate(
    n_panels: int,
    region: str = "Default",
    tilt_deg: float = 10
) -> Dict:
    """
    Quick energy and financial estimate for a given number of panels.
    Convenience function for rapid calculations.
    """
    calc = EnergyCalculator(region)
    calc.set_system(n_panels, tilt_deg)
    return calc.generate_full_report()


def compare_regions(n_panels: int) -> Dict[str, Dict]:
    """Compare energy yield across all Malaysian regions."""
    results = {}
    for region in MALAYSIA_IRRADIANCE.keys():
        if region == "Default":
            continue
        calc = EnergyCalculator(region)
        calc.set_system(n_panels)
        yield_data = calc.calculate_annual_yield()
        results[region] = {
            "annual_yield_kwh": yield_data["net_annual_yield_kwh"],
            "specific_yield": yield_data["specific_yield_kwh_kwp"],
            "daily_ghi": yield_data["daily_ghi_kwh_m2"],
        }
    return results
