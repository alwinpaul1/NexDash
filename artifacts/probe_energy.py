"""Decompose the energy model: ML vs physics, base vs gradient vs payload.
Run from repo root:  python artifacts/probe_energy.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexdash.physics import segment_energy_kwh
from nexdash.model import predict_energy
from nexdash.config import TRUCK

def ml(dist, pay, spd, grad, temp, wind=0.0):
    return predict_energy({"distance_km": dist, "payload_t": pay, "speed_kph": spd,
                           "gradient_pct": grad, "temperature_c": temp, "wind_mps": wind})

def ph(dist, pay, spd, grad, temp, wind=0.0):
    return segment_energy_kwh(dist, pay, spd, grad, temp, wind_mps=wind)

print(f"TRUCK: battery={TRUCK.battery_kwh} kerb={TRUCK.kerb_mass_kg} max_payload={TRUCK.max_payload_t}")
print("Note: payload_t=18 -> 18t kerb + 18t = 36t GCW;  payload_t=22 -> 40t GCW\n")

D = 100.0  # per-100km figures
print("=== kWh/100km, 83 km/h, 15 C, 0 wind (route-ish speed/temp) ===")
print(f"{'scenario':32} {'PHYSICS':>10} {'ML':>10}")
for pay in (0, 11, 18, 22):
    for grad in (0.0,):
        s = f"flat, payload={pay}t (GCW {18+pay}t)"
        print(f"{s:32} {ph(D,pay,83,grad,15):>10.1f} {ml(D,pay,83,grad,15):>10.1f}")
print()
print("=== gradient sweep, payload=18t (36t GCW), 83 km/h, 15 C ===")
for grad in (-6, -4, -2.5, -1, 0, 1, 2.5, 4, 6):
    print(f"grad={grad:+5.1f}%   physics={ph(D,18,83,grad,15):>8.1f}   ml={ml(D,18,83,grad,15):>8.1f}")
print()
print("=== speed sweep, payload=18t, flat, 15 C ===")
for spd in (60, 70, 80, 85, 89):
    print(f"speed={spd:>3} kph  physics={ph(D,18,spd,0,15):>8.1f}   ml={ml(D,18,spd,0,15):>8.1f}")
print()
print("=== temp sweep, payload=18t, flat, 83 km/h ===")
for temp in (-10, 0, 15, 20, 30):
    print(f"temp={temp:+4} C   physics={ph(D,18,83,0,temp):>8.1f}   ml={ml(D,18,83,0,temp):>8.1f}")
print()

# Single-call whole-route approximation: net gradient ~ flat
net_grad = (34 - 520) / (591 * 1000) * 100
print(f"=== Whole route as ONE segment (591 km, net grad {net_grad:+.3f}%, 18t, 83kph, 15C) ===")
print(f"  physics: {ph(591,18,83,net_grad,15):>8.1f} kWh  ({ph(591,18,83,net_grad,15)/591*100:.1f} /100km)")
print(f"  ml:      {ml(591,18,83,net_grad,15):>8.1f} kWh  ({ml(591,18,83,net_grad,15)/591*100:.1f} /100km)")
print()
print("Steady-state physics (this probe) ~122-126 /100km is the conservative basis the")
print("planner walks SOC + charging on. DISPLAYED energy now applies config.FIELD_CALIBRATION_FACTOR")
print("(0.85): the live Munich-Berlin headline is ~600 kWh = ~101.9 /100km (was 708 = 119.9).")
print("NexOS demo: 575 kWh = 94.6 /100km. Field band (REAL_WORLD_CALIBRATION.md S3): 95-105 /100km")
print("(ADAC 40t Munich-Woerth 88; Daimler tour 103). WLTP/spec end: 119-120 /100km.")
