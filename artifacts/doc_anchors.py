# Recompute the steady-state physics anchors quoted in README / REAL_WORLD_CALIBRATION
# at the CURRENT config (cd=0.50, CdA 5.0), so the docs cite real numbers, not cd=0.55-era.
import sys, dataclasses
sys.path.insert(0, "src")
from nexdash.config import TRUCK, FIELD_CALIBRATION_FACTOR

from nexdash.physics import segment_energy_kwh


def kwh_per_km(payload_t, speed, temp, cd=0.50, area=10.0, wind=0.0):
    truck = dataclasses.replace(TRUCK, cd=cd, frontal_area_m2=area)
    return segment_energy_kwh(distance_km=100.0, payload_t=payload_t, speed_kph=speed,
                              gradient_pct=0.0, temperature_c=temp, wind_mps=wind, truck=truck) / 100.0


print(f"config: cd={TRUCK.cd} area={TRUCK.frontal_area_m2} CdA={TRUCK.cd*TRUCK.frontal_area_m2:.2f} "
      f"crr={TRUCK.crr} dt={TRUCK.drivetrain_eff} | FIELD_CALIBRATION_FACTOR={FIELD_CALIBRATION_FACTOR}\n")
warm = kwh_per_km(22, 80, 20)
cold = kwh_per_km(22, 80, -10)
light_cold = kwh_per_km(4, 85, -10)
empty_warm = kwh_per_km(0, 80, 20)
print(f"40t / 80 / 20C  warm anchor : {warm:.3f} kWh/km   (was 1.265 @ cd0.55)")
print(f"40t / 80 / -10C cold        : {cold:.3f} kWh/km   (was 1.47  @ cd0.55)  swing +{(cold/warm-1)*100:.0f}%")
print(f"22t / 85 / -10C light-fast  : {light_cold:.3f} kWh/km   (was 1.55  @ cd0.55)")
print(f"18t / 80 / 20C  empty warm  : {empty_warm:.3f} kWh/km   (was ~0.90 @ cd0.55)")
print(f"\nfactor derivation: {FIELD_CALIBRATION_FACTOR} x warm {warm:.3f} = {FIELD_CALIBRATION_FACTOR*warm:.3f} kWh/km (field centre)")
print(f"empty->laden warm span: ~{empty_warm:.2f} -> {warm:.2f}")
