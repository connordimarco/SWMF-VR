"""Analytic dipole field-line geometry, used below the BATSRUS handoff radius.

Why this module exists
----------------------
The BATSRUS grid does not reach down to the ionosphere: inside rBody = 2.0 Re
the cells hold boundary-condition fill values, not solved physics, and tracing
through them draws field lines through the middle of the Earth. Gabor's
instruction (2026-09-01 email) is therefore:

  * from the ionosphere up to r = 2.5 Re, follow the ANALYTIC dipole
    (the run's tilted centered dipole B0 -- the first 3 IGRF harmonics;
    using full IGRF would be inconsistent with BATSRUS/RBE and is wrong),
  * from 2.5 Re outward, follow the numerically traced BATSRUS field
    (at 2.5 Re the full solution including B1 exists and the 1/8 Re
    resolution is acceptable).

The geometry of a dipole field line is simple in SM (Solar Magnetic)
coordinates, because the SM z axis IS the dipole axis. A field line with
L-shell parameter L obeys

    r = L * cos^2(lambda)

where r is the radius in Re, lambda is the magnetic latitude, and the SM
longitude is constant along the line (MLT is conserved). All the tilt
bookkeeping is handled by geopack's SM <-> GSM rotation, which must be set
for the frame time with geopack.recalc(ut) first (frames.recalc_for does
this).

Every function here works in degrees and Re, and says so in its names.
"""

import math
import os

from geopack import geopack

# Where the analytic dipole hands off to the numeric BATSRUS trace (Re).
# Decided by Gabor 2026-09-01: rBody is 2.0 and rCurrents is 2.375, but 2.5
# is where the full solution (B0 + B1) exists on a reasonably fine grid.
# Override with SWMF_HANDOFF_RE in .env: the first BATSRUS cell layer outside
# rBody carries body fill, so the handoff must sit a few cells above rBody
# (the April 2010 run has rBody = 2.5 with 1/8 Re cells -> 3.0).
HANDOFF_RADIUS_RE = float(os.environ.get('SWMF_HANDOFF_RE', '2.5'))

# Field lines start on the Earth's surface. The RBE foot-point latitudes are
# invariant latitudes referenced to r = 1 Re (checked 2026-08-31: the dipole
# rows of ro match 1/cos^2(lat) exactly), so this is the consistent choice.
FOOT_RADIUS_RE = 1.0


def l_shell_from_foot_latitude(foot_lat_deg):
    """L-shell of the dipole line whose foot point (at r = 1 Re) has the
    given magnetic latitude.   From r = L cos^2(lambda) with r = 1:
        L = 1 / cos^2(lambda_foot)
    """
    cos_lat = math.cos(math.radians(foot_lat_deg))
    return 1.0 / (cos_lat * cos_lat)


def latitude_where_line_crosses_radius(l_shell, radius_re):
    """Magnetic latitude (deg, positive) where the dipole line with this
    L-shell crosses the given radius.   From r = L cos^2(lambda):
        lambda = arccos( sqrt(r / L) )
    Only valid for radius_re <= L (the line's apex radius is L).
    """
    ratio = radius_re / l_shell
    if ratio > 1.0:
        raise ValueError('line with L = {:.3f} never reaches r = {:.3f} Re'
                         .format(l_shell, radius_re))
    return math.degrees(math.acos(math.sqrt(ratio)))


def sm_point_on_line(l_shell, mlt_hr, lat_deg):
    """SM cartesian coordinates (Re) of the point on the dipole line at the
    given magnetic latitude. SM longitude comes from MLT: midnight (MLT 0)
    is the -x direction, noon (MLT 12) is +x, so phi = (MLT - 12) * 15 deg.
    """
    lat_rad = math.radians(lat_deg)
    phi_rad = math.radians((mlt_hr - 12.0) * 15.0)
    radius_re = l_shell * math.cos(lat_rad) ** 2
    x_sm = radius_re * math.cos(lat_rad) * math.cos(phi_rad)
    y_sm = radius_re * math.cos(lat_rad) * math.sin(phi_rad)
    z_sm = radius_re * math.sin(lat_rad)
    return x_sm, y_sm, z_sm


def dipole_segment_gsm(l_shell, mlt_hr, lat_start_deg, lat_end_deg,
                       points_per_segment=40):
    """Sample the dipole line from lat_start to lat_end (deg, signed --
    positive is the northern hemisphere) and return a list of GSM (x, y, z)
    points in Re. Latitudes are sampled uniformly, which concentrates points
    near the foot where the line curves most -- that is what we want.

    geopack.recalc(ut) MUST already be set for this frame, because the
    SM -> GSM rotation depends on the dipole tilt at that instant.
    """
    points_gsm = []
    for step in range(points_per_segment):
        fraction = step / float(points_per_segment - 1)
        lat_deg = lat_start_deg + fraction * (lat_end_deg - lat_start_deg)
        x_sm, y_sm, z_sm = sm_point_on_line(l_shell, mlt_hr, lat_deg)
        x_gsm, y_gsm, z_gsm = geopack.smgsm(x_sm, y_sm, z_sm, 1)
        points_gsm.append((x_gsm, y_gsm, z_gsm))
    return points_gsm


def gsm_point_to_sm_lat_mlt_r(x_gsm, y_gsm, z_gsm):
    """Magnetic latitude (deg, signed), MLT (hours), and radius (Re) of a
    GSM point -- i.e. its SM spherical coordinates. Used to continue a
    numeric trace downward with the dipole from wherever it re-entered the
    handoff sphere.
    """
    x_sm, y_sm, z_sm = geopack.smgsm(x_gsm, y_gsm, z_gsm, -1)
    radius_re = math.sqrt(x_sm * x_sm + y_sm * y_sm + z_sm * z_sm)
    lat_deg = math.degrees(math.asin(z_sm / radius_re))
    phi_deg = math.degrees(math.atan2(y_sm, x_sm))
    mlt_hr = (phi_deg / 15.0 + 12.0) % 24.0
    return lat_deg, mlt_hr, radius_re


def north_segment_gsm(foot_lat_deg, mlt_hr,
                      handoff_radius_re=HANDOFF_RADIUS_RE,
                      points_per_segment=40):
    """The northern below-handoff piece of a field line: from the foot point
    on the Earth's surface down (in latitude) to where the line crosses the
    handoff sphere. Returns GSM points ordered foot -> handoff.
    """
    l_shell = l_shell_from_foot_latitude(foot_lat_deg)
    lat_at_handoff_deg = latitude_where_line_crosses_radius(
        l_shell, handoff_radius_re)
    return dipole_segment_gsm(l_shell, mlt_hr,
                              foot_lat_deg, lat_at_handoff_deg,
                              points_per_segment)


def south_segment_gsm_from_reentry(x_gsm, y_gsm, z_gsm,
                                   points_per_segment=40):
    """The below-handoff piece attached where the numeric trace re-entered
    the handoff sphere. The dipole line through the re-entry point has
        L = r / cos^2(lambda)
    and we follow it from the re-entry latitude to the Earth's surface,
    staying in the same hemisphere the trace actually ended in (no mirror
    assumption needed on this end). Returns GSM points ordered
    handoff -> foot.
    """
    lat_deg, mlt_hr, radius_re = gsm_point_to_sm_lat_mlt_r(x_gsm, y_gsm, z_gsm)
    cos_lat = math.cos(math.radians(lat_deg))
    l_shell = radius_re / (cos_lat * cos_lat)
    # Latitude magnitude where this line meets the Earth's surface (r = 1):
    foot_lat_mag_deg = latitude_where_line_crosses_radius(
        l_shell, FOOT_RADIUS_RE)
    hemisphere_sign = 1.0 if lat_deg >= 0.0 else -1.0
    return dipole_segment_gsm(l_shell, mlt_hr,
                              lat_deg, hemisphere_sign * foot_lat_mag_deg,
                              points_per_segment)


def full_dipole_line_gsm(foot_lat_deg, mlt_hr, points_per_segment=81):
    """A complete closed dipole line for L <= handoff radius: these lines
    never reach the BATSRUS trace region, so the whole line is analytic.
    Runs from the northern foot over the apex to the MIRROR southern foot --
    for these low, near-dipolar lines the perfect-conjugacy assumption
    (the same one RBE itself makes) is appropriate. Returns GSM points
    ordered north foot -> south foot.
    """
    l_shell = l_shell_from_foot_latitude(foot_lat_deg)
    return dipole_segment_gsm(l_shell, mlt_hr,
                              foot_lat_deg, -foot_lat_deg,
                              points_per_segment)
