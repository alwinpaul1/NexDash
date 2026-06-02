# Parametric aero/calibration probe for the eActros 600 energy model.
# Computes RAW steady-state kWh/100km (physics.segment_energy_kwh, no calibration)
# for a candidate drag coefficient, at 40 t and 36 t, flat, 80 km/h. Used to pin
# the right Cd + FIELD_CALIBRATION_FACTOR against real-world field figures.
#   python artifacts/aero_probe.py            # sweep table
#   python artifacts/aero_probe.py --cd 0.50  # single value (for agents)
import sys, argparse, dataclasses
sys.path.insert(0, "src")
from nexdash.config import TRUCK
from nexdash.physics import segment_energy_kwh


def raw_per_100km(cd, area, payload_t, speed=80.0, temp=15.0, wind=0.0):
    truck = dataclasses.replace(TRUCK, cd=cd, frontal_area_m2=area)
    return segment_energy_kwh(
        distance_km=100.0, payload_t=payload_t, speed_kph=speed,
        gradient_pct=0.0, temperature_c=temp, wind_mps=wind, truck=truck,
    )


ap = argparse.ArgumentParser()
ap.add_argument("--cd", type=float, default=None)
ap.add_argument("--area", type=float, default=TRUCK.frontal_area_m2)
args = ap.parse_args()

print(f"defaults: cd={TRUCK.cd} area={TRUCK.frontal_area_m2} CdA={TRUCK.cd*TRUCK.frontal_area_m2:.2f} "
      f"crr={TRUCK.crr} drivetrain={TRUCK.drivetrain_eff} regen={TRUCK.regen_eff} aux={TRUCK.aux_base_kw}kW")
# Warm anchor sanity: config docstring claims ~126.5 kWh/100km at 40t/80/20C.
anchor = raw_per_100km(TRUCK.cd, TRUCK.frontal_area_m2, 22.0, temp=20.0)
print(f"warm anchor (40t/80/20C, stock cd): raw={anchor:.1f} kWh/100km (docstring claims ~126.5)\n")

cds = [args.cd] if args.cd is not None else [0.40, 0.43, 0.45, 0.48, 0.50, 0.52, 0.55]
print(f"{'cd':>5} {'CdA':>5} | {'40t raw':>8} {'x.80':>6} {'x.85':>6} {'x.90':>6} | {'36t raw':>8} {'x.80':>6} {'x.85':>6}")
for cd in cds:
    r40 = raw_per_100km(cd, args.area, 22.0)
    r36 = raw_per_100km(cd, args.area, 18.0)
    print(f"{cd:>5.2f} {cd*args.area:>5.2f} | {r40:>8.1f} {r40*0.80:>6.1f} {r40*0.85:>6.1f} {r40*0.90:>6.1f} | "
          f"{r36:>8.1f} {r36*0.80:>6.1f} {r36*0.85:>6.1f}")
print("\nField anchors to hit (DISPLAY): ADAC 88 (40t efficient route), Daimler tour 103, hilly-laden ~105-115.")
