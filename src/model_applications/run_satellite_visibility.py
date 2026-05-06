#!/usr/bin/env python3
"""
run_satellite_visibility.py — Satellite visibility analysis with GAIA DR3 cross-match.

Strategy:
  1. PRIMARY  — read rdls.fits (RA/Dec) → query GAIA DR3 → cross-match with AXY objects only
  2. FALLBACK — if no rdls.fits, try WCS from -image.fits → query GAIA DR3 → cross-match with AXY
  3. SKIP     — if neither works, skip the entry

GAIA queries are cached to disk so re-runs are instant.

Usage:
    python -m src.model_applications.run_satellite_visibility
        --model /app/data/trained-models/model.pt
        --dataset /app/data/dataset/
        --output /app/data/performance/
        --sat-mag 4.2
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia

sys.path.append(str(Path(__file__).parent.parent))

from model_training.data_preprocessing.loader import DatasetLoader
from model_training.data_preprocessing.normalization.log_normalization import LogPercentileNormalization
from model_training.data_preprocessing.masking.circular_dynamic_masking import CircularDynamicMasking
from model_analysis.detector import Detector
from model_analysis.postprocessing.morphological_postprocessing import MorphologicalClosing


# ---------------------------------------------------------------------------
# Physical constants — TeideSat I document (Javier Marrero García, 2022)
# ---------------------------------------------------------------------------

VEGA_FLUX_R_BAND_ERG     = 2.16e-9
R_BAND_WIDTH_ANGSTROM    = 1600.0
SENSITIVITY_W_PER_SR     = 1.0 / 972.2
LED_AREA_CM2             = 11.3
ERG_PER_WATT             = 1e7
SOLID_ANGLE_SR           = 4 * np.pi

GAIA_MAG_COL             = "phot_rp_mean_mag"
GAIA_MAG_COL_FALLBACK    = "phot_g_mean_mag"
CROSSMATCH_RADIUS_ARCSEC = 5.0
MIN_CALIBRATED_OBJECTS   = 30


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MagRecord:
    gaia_mag:  float
    detected:  bool
    flux_adu:  float
    source:    str


@dataclass
class CompletenessCurve:
    bin_centers:  np.ndarray
    completeness: np.ndarray
    counts:       np.ndarray
    mag_50pct:    float
    mag_80pct:    float


@dataclass
class SatelliteParams:
    apparent_magnitude: Optional[float] = None
    flux_erg:           Optional[float] = None
    led_lumens:         Optional[float] = None
    num_leds:           int             = 4
    label:              str             = "TeideSat I"


def lumens_to_flux_erg(lumens_per_led: float, num_leds: int) -> float:
    total_lm     = lumens_per_led * num_leds
    power_w      = total_lm * SENSITIVITY_W_PER_SR * SOLID_ANGLE_SR
    flux_w_cm2   = power_w / LED_AREA_CM2
    flux_erg_cm2 = flux_w_cm2 * ERG_PER_WATT
    return flux_erg_cm2 / R_BAND_WIDTH_ANGSTROM


def flux_erg_to_mag(flux_erg: float) -> float:
    if flux_erg <= 0:
        return np.inf
    return -2.5 * np.log10(flux_erg / VEGA_FLUX_R_BAND_ERG)


def mag_to_flux_erg(mag: float) -> float:
    return VEGA_FLUX_R_BAND_ERG * 10 ** (-0.4 * mag)


def cache_path(cache_dir: Path, entry_id: str) -> Path:
    return cache_dir / f"{entry_id}.json"


def load_cache(cache_dir: Path, entry_id: str) -> Optional[list[dict]]:
    p = cache_path(cache_dir, entry_id)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def save_cache(cache_dir: Path, entry_id: str, data: list[dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path(cache_dir, entry_id), "w") as f:
        json.dump(data, f)


def query_gaia_cone(centre: SkyCoord, radius_arcsec: float) -> Optional[object]:
    try:
        Gaia.ROW_LIMIT = 5000
        job = Gaia.cone_search_async(
            coordinate = centre,
            radius     = radius_arcsec * u.arcsec,
            table_name = "gaiadr3.gaia_source",
            columns    = ["source_id", "ra", "dec",
                          GAIA_MAG_COL, GAIA_MAG_COL_FALLBACK],
            verbose    = False,
        )
        table = job.get_results()
        return table if table is not None and len(table) > 0 else None
    except Exception as exc:
        print(f"GAIA query failed: {exc}")
        return None


def extract_mag(row) -> Optional[float]:
    for col in (GAIA_MAG_COL, GAIA_MAG_COL_FALLBACK):
        try:
            val = float(row[col])
            if np.isfinite(val):
                return val
        except Exception:
            continue
    return None


def gaia_table_to_list(table) -> list[dict]:
    """Convert astropy Table to plain list of dicts for JSON serialisation."""
    result = []
    for row in table:
        mag = extract_mag(row)
        if mag is None:
            continue
        result.append({
            "ra":  float(row["ra"]),
            "dec": float(row["dec"]),
            "mag": mag,
        })
    return result


def load_wcs(entry_dir: Path, entry_id: str) -> Optional[WCS]:
    candidates = [
        entry_dir / f"{entry_id}-wcs.fits",
        entry_dir / f"{entry_id}-image.fits",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with fits.open(path) as hdul:
                for hdu in hdul:
                    hdr = hdu.header
                    if "CTYPE1" in hdr and "CRVAL1" in hdr:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            wcs = WCS(hdr, naxis=2)
                        ra, dec = wcs.wcs.crval
                        if np.isfinite(ra) and np.isfinite(dec):
                            return wcs
        except Exception:
            continue
    return None


def pixel_scale_arcsec(wcs: WCS) -> float:
    try:
        scales = wcs.proj_plane_pixel_scales()
        return float(np.mean([s.to(u.arcsec).value for s in scales]))
    except Exception:
        return 1.0


def get_gaia_sources(
    entry_dir:   Path,
    entry_id:    str,
    image_shape: tuple,
    cache_dir:   Path,
) -> tuple[Optional[list[dict]], Optional[WCS], str]:
    """
    Returns (gaia_sources, wcs, strategy) where strategy is 'rdls', 'wcs', or 'skip'.
    gaia_sources is a list of dicts: {ra, dec, mag}
    """
    cached = load_cache(cache_dir, entry_id)
    if cached is not None:
        wcs = load_wcs(entry_dir, entry_id)
        strategy = cached[0].get("_strategy", "rdls") if cached else "rdls"
        sources = [s for s in cached if "_strategy" not in s]
        return sources, wcs, f"{strategy}(cached)"

    rdls_path = entry_dir / "rdls.fits"
    if rdls_path.exists():
        try:
            with fits.open(rdls_path) as hdul:
                data = hdul[1].data
                if data is not None and len(data) > 0:
                    ra  = np.array(data["RA"],  dtype=float)
                    dec = np.array(data["DEC"], dtype=float)
                    valid = np.isfinite(ra) & np.isfinite(dec)
                    if valid.sum() > 0:
                        rdls_coords = SkyCoord(ra=ra[valid]*u.deg, dec=dec[valid]*u.deg)
                        centre      = SkyCoord(
                            ra=rdls_coords.ra.mean(), dec=rdls_coords.dec.mean()
                        )
                        seps    = centre.separation(rdls_coords).arcsec
                        radius  = seps.max() + 60.0
                        table   = query_gaia_cone(centre, radius)
                        if table is not None:
                            sources = gaia_table_to_list(table)
                            save_cache(cache_dir, entry_id,
                                       [{"_strategy": "rdls"}] + sources)
                            wcs = load_wcs(entry_dir, entry_id)
                            return sources, wcs, "rdls"
        except Exception:
            pass

    wcs = load_wcs(entry_dir, entry_id)
    if wcs is not None:
        h, w = image_shape
        try:
            centre    = wcs.pixel_to_world(w / 2.0, h / 2.0)
            pix_scale = pixel_scale_arcsec(wcs)
            radius    = 0.5 * np.sqrt(w**2 + h**2) * pix_scale * 1.2
            table     = query_gaia_cone(centre, radius)
            if table is not None:
                sources = gaia_table_to_list(table)
                save_cache(cache_dir, entry_id,
                           [{"_strategy": "wcs"}] + sources)
                return sources, wcs, "wcs"
        except Exception:
            pass

    save_cache(cache_dir, entry_id, [{"_strategy": "skip"}])
    return None, wcs, "skip"


def crossmatch_axy_to_gaia(
    axy_objects:  list,          
    gaia_sources: list[dict],   
    wcs:          WCS,
    radius_arcsec: float = CROSSMATCH_RADIUS_ARCSEC,
) -> list[dict]:
    """
    For each AXY object:
      1. Project pixel → sky via WCS
      2. Find nearest GAIA source within radius_arcsec
      3. Assign that GAIA magnitude to the AXY object

    Returns list of dicts: {x, y, flux_adu, gaia_mag}
    Only AXY objects that have a GAIA match are returned.
    """
    if not axy_objects or not gaia_sources:
        return []

    axy_px = np.array([(o[0], o[1]) for o in axy_objects])
    try:
        axy_sky = wcs.pixel_to_world(axy_px[:, 0], axy_px[:, 1])
    except Exception:
        return []

    axy_coords  = SkyCoord(ra=axy_sky.ra, dec=axy_sky.dec)
    gaia_coords = SkyCoord(
        ra  = [s["ra"]  for s in gaia_sources] * u.deg,
        dec = [s["dec"] for s in gaia_sources] * u.deg,
    )

    idx, sep2d, _ = axy_coords.match_to_catalog_sky(gaia_coords)

    matched = []
    for i, (axy_obj, gi, sep) in enumerate(zip(axy_objects, idx, sep2d.arcsec)):
        if sep > radius_arcsec:
            continue
        matched.append({
            "x":        float(axy_obj[0]),
            "y":        float(axy_obj[1]),
            "flux_adu": float(axy_obj[2]),
            "gaia_mag": gaia_sources[gi]["mag"],
        })

    return matched


def collect_mag_records(
    test_dataset: dict,
    dataset_path: Path,
    detector:     Detector,
    cache_dir:    Path,
    tolerance_px: int   = 5,
    max_entries:  Optional[int] = None,
    crossmatch_r: float = CROSSMATCH_RADIUS_ARCSEC,
) -> list[MagRecord]:

    records: list[MagRecord] = []
    entries_list = list(test_dataset.items())
    if max_entries:
        entries_list = entries_list[:max_entries]

    n_total  = len(entries_list)
    n_rdls   = 0
    n_wcs    = 0
    n_skip   = 0

    for i, (entry_id, entry) in enumerate(entries_list):
        print(f"  [{i+1:>3}/{n_total}] {entry_id}", end=" ", flush=True)

        entry_dir   = dataset_path / entry_id
        image_shape = entry.nn_input_image.shape[:2]
        gt_objects  = entry.filtered_objects   # [(x, y, flux), ...]

        if not gt_objects:
            print("[skip] no GT objects")
            n_skip += 1
            continue

        gaia_sources, wcs, strategy = get_gaia_sources(
            entry_dir, entry_id, image_shape, cache_dir
        )

        if gaia_sources is None or wcs is None:
            print(f"[skip] {strategy}")
            n_skip += 1
            continue

        matched = crossmatch_axy_to_gaia(gt_objects, gaia_sources, wcs, crossmatch_r)
        if not matched:
            print(f"[skip] 0/{len(gt_objects)} AXY objects matched to GAIA")
            n_skip += 1
            continue

        mask     = detector.predict_mask(entry.nn_input_image)
        pred_pos = detector.postprocessing.extract_positions(mask)
        pred_arr = np.array(pred_pos) if pred_pos else np.empty((0, 2))

        for obj in matched:
            gx, gy = obj["x"], obj["y"]
            if pred_arr.shape[0] == 0:
                detected = False
            else:
                dists    = np.sqrt((pred_arr[:, 0]-gx)**2 + (pred_arr[:, 1]-gy)**2)
                detected = bool(dists.min() <= tolerance_px)
            records.append(MagRecord(
                gaia_mag = obj["gaia_mag"],
                detected = detected,
                flux_adu = obj["flux_adu"],
                source   = strategy.replace("(cached)", ""),
            ))

        is_cached = "(cached)" in strategy
        tag       = strategy
        n_det     = sum(r.detected for r in records[-len(matched):])
        print(f"[{tag}] {len(matched)}/{len(gt_objects)} matched, {n_det}/{len(matched)} detected")

        if "rdls" in strategy:
            n_rdls += 1
        elif "wcs" in strategy:
            n_wcs += 1

    print(f"\n  rdls strategy : {n_rdls}/{n_total}")
    print(f"  wcs  strategy : {n_wcs}/{n_total}")
    print(f"  skipped       : {n_skip}/{n_total}")
    print(f"  Total calibrated objects : {len(records)}")
    return records


def compute_completeness(
    records:     list[MagRecord],
    n_bins:      int = 20,
    min_per_bin: int = 5,
) -> CompletenessCurve:
    mags     = np.array([r.gaia_mag for r in records])
    detected = np.array([r.detected for r in records], dtype=float)

    edges   = np.linspace(mags.min(), mags.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    completeness = np.full(n_bins, np.nan)
    counts       = np.zeros(n_bins, dtype=int)

    for k in range(n_bins):
        mask      = (mags >= edges[k]) & (mags < edges[k + 1])
        counts[k] = mask.sum()
        if counts[k] >= min_per_bin:
            completeness[k] = detected[mask].mean()

    def mag_at_completeness(target: float) -> float:
        valid_idx = np.where(~np.isnan(completeness))[0]
        if len(valid_idx) < 2:
            return np.nan
        x = centers[valid_idx]
        y = completeness[valid_idx]
        for j in range(len(x) - 1):
            if y[j] >= target >= y[j + 1]:
                t = (target - y[j]) / (y[j + 1] - y[j] + 1e-12)
                return x[j] + t * (x[j + 1] - x[j])
        return x[-1]

    return CompletenessCurve(
        bin_centers  = centers,
        completeness = completeness,
        counts       = counts,
        mag_50pct    = mag_at_completeness(0.50),
        mag_80pct    = mag_at_completeness(0.80),
    )


def plot_results(
    curve:      CompletenessCurve,
    sat:        SatelliteParams,
    records:    list[MagRecord],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Detection Limit — GAIA-Calibrated (AXY-only cross-match)", fontsize=13)

    ax = axes[0]
    valid = ~np.isnan(curve.completeness)
    ax.plot(curve.bin_centers[valid], curve.completeness[valid], "o-",
            color="#378ADD", lw=2, ms=5, label="Completeness")
    ax.axhline(0.5, color="#888780", lw=1, ls="--", alpha=0.8)
    ax.axhline(0.8, color="#888780", lw=1, ls="-.", alpha=0.8)
    ax.text(curve.bin_centers[valid].min() + 0.1, 0.51, "50%", color="#888780", fontsize=9)
    ax.text(curve.bin_centers[valid].min() + 0.1, 0.81, "80%", color="#888780", fontsize=9)

    if np.isfinite(curve.mag_50pct):
        ax.axvline(curve.mag_50pct, color="#E24B4A", lw=1.5, ls="--",
                   label=f"50% limit = {curve.mag_50pct:.2f} mag")
    if np.isfinite(curve.mag_80pct):
        ax.axvline(curve.mag_80pct, color="#EF9F27", lw=1.5, ls="--",
                   label=f"80% limit = {curve.mag_80pct:.2f} mag")
    if sat.apparent_magnitude is not None:
        ax.axvline(sat.apparent_magnitude, color="#1D9E75", lw=2,
                   label=f"{sat.label} = {sat.apparent_magnitude:.1f} mag")
        ax.text(sat.apparent_magnitude + 0.05, 0.05, sat.label,
                color="#1D9E75", fontsize=9, rotation=90, va="bottom")

    ax.set_xlabel("GAIA magnitude (RP ≈ Cousins R)")
    ax.set_ylabel("Detection completeness")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Completeness vs. real magnitude\n(only AXY-catalogue objects)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(curve.bin_centers, curve.counts,
            width=np.diff(curve.bin_centers).mean() * 0.8,
            color="#378ADD", alpha=0.7)
    ax2.set_xlabel("GAIA magnitude")
    ax2.set_ylabel("AXY object count")
    ax2.set_title("AXY objects per magnitude bin")
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[2]
    mags_det  = [r.gaia_mag for r in records if r.detected]
    mags_miss = [r.gaia_mag for r in records if not r.detected]
    all_mags  = [r.gaia_mag for r in records]
    bins      = np.linspace(min(all_mags), max(all_mags), 30)
    ax3.hist(mags_det,  bins=bins, color="#378ADD", alpha=0.75, label="Detected")
    ax3.hist(mags_miss, bins=bins, color="#E24B4A", alpha=0.75, label="Missed")
    if sat.apparent_magnitude is not None:
        ax3.axvline(sat.apparent_magnitude, color="#1D9E75", lw=2, label=f"{sat.label}")
    ax3.set_xlabel("GAIA magnitude")
    ax3.set_ylabel("Count")
    ax3.set_title("Detected vs. missed by magnitude\n(AXY objects only)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "satellite_visibility.png"
    plt.savefig(str(out), dpi=200)
    plt.show()
    print(f"Plot saved: {out}")


def print_verdict(curve: CompletenessCurve, sat: SatelliteParams) -> None:
    sep = "─" * 64
    print(f"\n{sep}")
    print("  SATELLITE VISIBILITY ANALYSIS  —  GAIA-calibrated (AXY-only)")
    print(sep)

    if np.isfinite(curve.mag_50pct):
        print(f"  Model 50% detection limit : {curve.mag_50pct:.2f} mag")
    else:
        print("  Model 50% detection limit : undetermined")
    if np.isfinite(curve.mag_80pct):
        print(f"  Model 80% detection limit : {curve.mag_80pct:.2f} mag")

    print()

    if sat.apparent_magnitude is None:
        print("  No satellite data provided.")
        print("  Re-run with --sat-mag <mag>  or  --led-lumens <lm> --num-leds <n>")
        print("  TeideSat I document value: --sat-mag 4.2  (range 4.0–4.5)")
        print(sep)
        return

    print(f"  {sat.label} apparent magnitude : {sat.apparent_magnitude:.2f} mag")

    if not np.isfinite(curve.mag_50pct):
        print("\n  Cannot emit verdict — detection limit undetermined.")
        print(sep)
        return

    margin = curve.mag_50pct - sat.apparent_magnitude
    print(f"\n  Margin vs 50% limit : {margin:+.2f} mag  (positive = detectable)")
    print()

    if margin > 2.0:
        print(f"  ✓✓  VERY LIKELY DETECTABLE")
        print(f"      {sat.label} ({sat.apparent_magnitude:.1f} mag) is {margin:.1f} mag")
        print(f"      brighter than the model 50% limit ({curve.mag_50pct:.2f} mag).")
    elif margin > 0.5:
        print(f"  ✓   LIKELY DETECTABLE")
        print(f"      {sat.label} is {margin:.1f} mag above the 50% limit.")
    elif margin > -0.5:
        print(f"  ?   UNCERTAIN — near detection boundary ({margin:+.2f} mag)")
    else:
        print(f"  ✗   LIKELY NOT DETECTABLE")
        print(f"      {sat.label} ({sat.apparent_magnitude:.1f} mag) is {abs(margin):.1f} mag")
        print(f"      below the model 50% limit ({curve.mag_50pct:.2f} mag).")

    print(f"\n{sep}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run_satellite_visibility.py",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",             required=True,  type=Path)
    p.add_argument("--dataset",           required=True,  type=Path)
    p.add_argument("--output",            default=Path("../../data/performance/"), type=Path)
    p.add_argument("--cache-dir",         default=None,   type=Path,
                   help="Directory for GAIA query cache. Default: <output>/gaia_cache/")
    p.add_argument("--shape",             default=256,    type=int)
    p.add_argument("--tolerance-px",      default=5,      type=int)
    p.add_argument("--bins",              default=20,     type=int)
    p.add_argument("--crossmatch-radius", default=CROSSMATCH_RADIUS_ARCSEC, type=float)
    p.add_argument("--max-entries",       default=None,   type=int,
                   help="Limit entries processed (useful for quick tests)")
    p.add_argument("--sat-mag",           type=float,     default=None,
                   help="Satellite apparent magnitude (Vega, Cousins R). TeideSat I: 4.2")
    p.add_argument("--led-lumens",        type=float,     default=None,
                   help="Lumens per LED → automatic magnitude conversion")
    p.add_argument("--num-leds",          type=int,       default=4)
    p.add_argument("--sat-label",         type=str,       default="TeideSat I")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output / "gaia_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"GAIA cache directory: {cache_dir}")

    target_shape   = (args.shape, args.shape)
    normalization  = LogPercentileNormalization()
    masking        = CircularDynamicMasking()
    postprocessing = MorphologicalClosing()

    loader  = DatasetLoader(normalization=normalization, masking=masking, target_shape=target_shape)
    print("Loading dataset...")
    dataset = loader.load(args.dataset)

    items = list(dataset.items())
    _, rest       = train_test_split(items, train_size=0.5, shuffle=True, random_state=42)
    test_items, _ = train_test_split(rest,  train_size=0.5, shuffle=True, random_state=42)
    test_dataset  = dict(test_items)
    print(f"Test set: {len(test_dataset)} entries")

    detector = Detector.from_saved_model(
        model_path     = args.model,
        postprocessing = postprocessing,
        normalization  = normalization,
        target_shape   = target_shape,
    )

    print(f"\nProcessing entries (GAIA cache: {cache_dir})...")
    records = collect_mag_records(
        test_dataset = test_dataset,
        dataset_path = args.dataset,
        detector     = detector,
        cache_dir    = cache_dir,
        tolerance_px = args.tolerance_px,
        max_entries  = args.max_entries,
        crossmatch_r = args.crossmatch_radius,
    )

    if len(records) < MIN_CALIBRATED_OBJECTS:
        print(f"\nWARNING: only {len(records)} calibrated objects "
              f"(recommended ≥ {MIN_CALIBRATED_OBJECTS}).")

    if len(records) == 0:
        print("\nNo calibrated objects. Check rdls.fits files and internet connection.")
        sys.exit(1)

    print(f"\nTotal calibrated objects : {len(records)}")
    print(f"  via rdls : {sum(1 for r in records if r.source == 'rdls')}")
    print(f"  via wcs  : {sum(1 for r in records if r.source == 'wcs')}")
    print(f"  Detected : {sum(r.detected for r in records)}")
    print(f"  Missed   : {sum(not r.detected for r in records)}")

    curve = compute_completeness(records, n_bins=args.bins)

    sat = SatelliteParams(label=args.sat_label, num_leds=args.num_leds)
    if args.sat_mag is not None:
        sat.apparent_magnitude = args.sat_mag
        sat.flux_erg           = mag_to_flux_erg(args.sat_mag)
    elif args.led_lumens is not None:
        sat.flux_erg           = lumens_to_flux_erg(args.led_lumens, args.num_leds)
        sat.apparent_magnitude = flux_erg_to_mag(sat.flux_erg)
        sat.led_lumens         = args.led_lumens
        print(f"\nConverted {args.num_leds}×{args.led_lumens:.0f} lm "
              f"→ mag {sat.apparent_magnitude:.2f} (Cousins R, Vega)")

    plot_results(curve, sat, records, args.output)
    print_verdict(curve, sat)


if __name__ == "__main__":
    main()