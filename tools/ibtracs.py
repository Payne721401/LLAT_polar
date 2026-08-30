"""IBTrACS as the best-track source, matched to this project's storm ids.

The paper verifies against IBTrACS position and MSLP and states that each
verified sample corresponds to a best-track record. Numbers meant to stand
beside the paper's have to come from the same archive, so this is the source
for both truncation and best-track truth. The JMA lists under
ERA5_2024_for_TC/TC_list_JMA_v2 end within 0-18 h of IBTrACS and were used
first, but "close enough" is not "the same", and JMA's winds are 10-minute
against IBTrACS/JTWC's 1-minute, roughly 12 % apart and more at the top end.

**The storm numbers are not interchangeable.** JMA numbers named storms; JTWC
numbers every depression it tracks. The two run out of step partway through a
season, so 202405W is MARIA while WP052024 is GAEMI, and mapping by number
produces a table of nonsense that looks plausible. Storms are matched here by
where and when they were: the initial time of a case and the centre of its ERA5
box identify exactly one IBTrACS storm.

One file holds every storm in the basin, so it is parsed once per process and
indexed by year. The WP file is about 110 MB and takes a few seconds; that cost
is paid once, before any thread pool starts.

Download:
    wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-\\
climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv
"""
import csv
import datetime
import math
import os

# One archive, one parse, keyed by the file it came from.
_CACHE = {}
_MATCH = {}

DEG_KM = 111.32


def _f(row, *names):
    """First of `names` that parses as a float, or None."""
    for n in names:
        v = (row.get(n) or "").strip()
        if v:
            try:
                return float(v)
            except ValueError:
                pass
    return None


def load(path):
    """{sid: {"name", "atcf", "rec": {datetime: {...}}}} for one IBTrACS file.

    Columns are taken in the order the paper's truth would: LAT/LON are the
    archive's merged position; pressure prefers the WMO agency (RSMC Tokyo in
    this basin) and falls back to USA; wind prefers USA, because the
    Saffir-Simpson categories and the paper's 65 kt threshold are defined on
    1-minute winds and WMO_WIND here is 10-minute.
    """
    path = os.path.expanduser(path)
    if path in _CACHE:
        return _CACHE[path]
    out = {}
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = csv.DictReader(fh)
        for i, r in enumerate(rows):
            if i == 0:          # the units row, not data
                continue
            t = (r.get("ISO_TIME") or "").strip()
            lat = _f(r, "LAT", "USA_LAT", "TOKYO_LAT")
            lon = _f(r, "LON", "USA_LON", "TOKYO_LON")
            if not t or lat is None or lon is None:
                continue
            try:
                when = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            d = out.setdefault(r["SID"], {"name": "", "atcf": "", "rec": {}})
            nm = (r.get("NAME") or "").strip()
            if nm and nm != "UNNAMED":
                d["name"] = nm
            a = (r.get("USA_ATCF_ID") or "").strip()
            if a:
                d["atcf"] = a
            d["rec"][when] = {
                "lon": lon % 360.0, "lat": lat,
                "pres": _f(r, "WMO_PRES", "USA_PRES", "TOKYO_PRES"),
                "vmax": _f(r, "USA_WIND", "WMO_WIND", "TOKYO_WIND"),
                "nature": (r.get("NATURE") or "").strip(),
            }
    _CACHE[path] = out
    return out


def km(dlon, dlat, lat):
    """Great-circle-ish separation, the same approximation track_error uses."""
    return math.hypot(dlon * DEG_KM * math.cos(math.radians(lat)),
                      dlat * DEG_KM)


def match(path, tc_id, when, lon, lat, max_km=400.0, max_hours=6.0):
    """The IBTrACS storm that was at (lon, lat) at `when`, or None.

    tc_id is only a cache key and a name for the error; the identification is
    entirely by position and time, so no assumption about how this project's
    numbers relate to anyone else's can leak in.

    max_km is generous on purpose. The position being matched is the centre of
    an ERA5 box, which sits within about 30 km of its own pressure minimum, and
    the archive is 6-hourly while a case can start between records; 400 km is
    far tighter than the separation between two simultaneous storms in this
    basin and far looser than either of those errors.
    """
    key = (path, tc_id)
    if key in _MATCH:
        return _MATCH[key]
    best, best_km = None, None
    for sid, d in load(path).items():
        for t, rec in d["rec"].items():
            dt = abs((t - when).total_seconds()) / 3600.0
            if dt > max_hours:
                continue
            sep = km(rec["lon"] - (lon % 360.0), rec["lat"] - lat, lat)
            if sep <= max_km and (best_km is None or sep < best_km):
                best, best_km = sid, sep
    _MATCH[key] = best
    return best


def times(path, sid):
    """Set of valid times with a record - what a lead is clipped against."""
    d = load(path).get(sid)
    return set(d["rec"]) if d else set()


def positions(path, sid):
    """{datetime: (lon, lat)} for use as verification truth."""
    d = load(path).get(sid)
    return {t: (r["lon"], r["lat"]) for t, r in d["rec"].items()} if d else {}


def peak_vmax(path, sid):
    """Highest 1-minute wind over the storm's life, in kt, or None."""
    d = load(path).get(sid)
    v = [r["vmax"] for r in d["rec"].values() if r["vmax"] is not None] if d else []
    return max(v) if v else None
