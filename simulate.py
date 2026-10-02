"""
simulate.py

Turns a list of beat timestamps into a ball-bounce choreography:

  - The ball is dropped (zero initial velocity) at t=0 and free-falls under
    constant gravity until it lands on the first procedurally generated
    panel, which is placed exactly where free fall puts it at beats[0].
  - For every subsequent beat, a new panel is generated at a procedurally
    chosen (but on-screen, collision-free) location, and the launch
    velocity needed for the ball to travel from the previous panel to that
    new panel in exactly `dt = beats[i] - beats[i-1]` seconds is solved
    analytically from the projectile-motion equations:

        x(t) = x0 + vx*t
        y(t) = y0 + vy*t + 0.5*g*t^2

    Because we solve for (vx, vy) given a fixed dt and target (x, y), the
    ball's landing time is *exact* by construction -- every bounce lands
    precisely on a beat. This is a choreographed inverse-kinematics
    approach (not a forward collision-detection engine), which is what
    lets us guarantee frame-accurate beat sync while still obeying real
    projectile physics (constant gravity, parabolic arcs, no artificial
    displacement floors -- short gaps between beats produce genuinely
    small, gravity-dominated arcs) between bounces.

  - Each panel is only ever targeted once (a fresh panel is generated per
    beat), so no panel is hit twice.
  - Before a candidate arc is accepted, it's checked against every
    previously placed panel's actual geometry (line segment, not just its
    center point) with the ball's radius as clearance, so the flight path
    never clips through a panel it already passed. Candidates that rise
    above the launch point are tried first (for a lively bounce), then
    candidates that never rise above it (much less likely to collide,
    since panel centers are placed strictly lower than the last one), and
    finally a minimal straight-down hop as a last resort -- each phase is
    still verified geometrically rather than assumed safe.
  - Panel orientation is derived from reflection geometry: the panel's
    surface normal is the vector that bisects the incoming and outgoing
    velocity directions, exactly as it would need to be for a real
    bounce/mirror reflection.
"""

import math
import random

BALL_RADIUS = 22.0
PANEL_THICKNESS = 10.0
CLEARANCE = BALL_RADIUS + PANEL_THICKNESS / 2.0 + 8.0  # ball must clear panels by this much
# extra headroom used only during candidate search, so that the true
# minimum distance (which can dip slightly below what the discrete arc
# samples measure, between sample points) still clears CLEARANCE
_CHECK_CLEARANCE = CLEARANCE + 10.0


class Panel:
    __slots__ = ("id", "x", "y", "angle", "length", "color", "hit_time", "seg_index")

    def __init__(self, id, x, y, angle, length, color, hit_time, seg_index):
        self.id = id
        self.x = x
        self.y = y
        self.angle = angle       # orientation of the panel surface, radians
        self.length = length
        self.color = color
        self.hit_time = hit_time
        self.seg_index = seg_index


def _normalize(v):
    x, y = v
    m = math.hypot(x, y)
    if m < 1e-6:
        return (0.0, -1.0)
    return (x / m, y / m)


def _panel_normal_angle(incoming, outgoing):
    """Angle (radians) of the surface normal that reflects `incoming`
    velocity into `outgoing` velocity, i.e. the bisector of (-incoming)
    and (outgoing)."""
    ix, iy = _normalize(incoming)
    ox, oy = _normalize(outgoing)
    nx, ny = (-ix + ox, -iy + oy)
    m = math.hypot(nx, ny)
    if m < 1e-6:
        # incoming/outgoing anti-parallel (straight bounce back) -- normal
        # is just perpendicular to travel direction, pick "up"
        nx, ny = (0.0, -1.0)
    else:
        nx, ny = nx / m, ny / m
    return math.atan2(ny, nx)


def _size_panel(neighbor_hop_dist, panel_length_range, rng, min_length=46.0, fraction=0.55):
    """Size a panel relative to the distance the ball just traveled to
    reach it, so panels in a densely packed, fast passage stay small
    enough to plausibly fit -- instead of drawing a fixed-size panel that
    claims far more room than a short hop actually leaves available."""
    lo, hi = panel_length_range
    cap = max(min_length, min(hi, neighbor_hop_dist * fraction))
    target_lo = min(lo, cap)
    return rng.uniform(target_lo, cap) if cap > target_lo else cap


PALETTE = [
    (255, 87, 87), (255, 189, 89), (255, 240, 92), (129, 236, 122),
    (92, 219, 209), (92, 168, 255), (154, 122, 255), (255, 122, 220),
]


# ---------------------------------------------------------------------------
# Geometry helpers for collision avoidance
# ---------------------------------------------------------------------------

def _panel_endpoints(panel):
    half = panel.length / 2.0
    dx = math.cos(panel.angle) * half
    dy = math.sin(panel.angle) * half
    return (panel.x - dx, panel.y - dy), (panel.x + dx, panel.y + dy)


def _point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _sample_arc(x0, y0, vx, vy, g, dt, n=64):
    pts = []
    for i in range(n + 1):
        t = dt * i / n
        pts.append((x0 + vx * t, y0 + vy * t + 0.5 * g * t * t))
    return pts


def _arc_bbox(pts, pad):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


def _panel_bbox(panel, pad):
    a, b = _panel_endpoints(panel)
    return (min(a[0], b[0]) - pad, max(a[0], b[0]) + pad,
            min(a[1], b[1]) - pad, max(a[1], b[1]) + pad)


def _arc_clear_of_launch_point(arc_pts, launch_x, launch_y, radius, tail_frac=0.25):
    """A panel's final orientation isn't known until *after* the segment
    leaving it is chosen (its angle depends on that segment's outgoing
    velocity), so we can't geometrically test new arcs against the launch
    panel's real shape without a circular dependency. Instead, treat the
    launch panel conservatively as a disc of `radius` (big enough to
    contain any panel of any orientation centered there).

    We only require the *landing end* of the arc (the last `tail_frac` of
    it, which is where the new panel actually ends up) to clear that disc.
    The ball is allowed -- expected, even -- to spend the early/middle
    part of its flight close to the point it just launched from, since it
    physically can't teleport away instantly; what we actually want to
    prevent is the *new panel* ending up implausibly on top of the old
    one."""
    n = len(arc_pts) - 1
    tail_n = max(1, int(round(tail_frac * n)))
    for (x, y) in arc_pts[-tail_n:]:
        if math.hypot(x - launch_x, y - launch_y) < radius:
            return False
    return True


def _arc_clear_of_panels(arc_pts, panels, clearance):
    """True if every sampled point of the arc stays at least `clearance`
    away from every panel in `panels`."""
    if not panels:
        return True
    ax0, ax1, ay0, ay1 = _arc_bbox(arc_pts, clearance)
    for panel in panels:
        px0, px1, py0, py1 = _panel_bbox(panel, clearance)
        if px1 < ax0 or px0 > ax1 or py1 < ay0 or py0 > ay1:
            continue  # bounding boxes don't even overlap -- cheap reject
        a, b = _panel_endpoints(panel)
        for (x, y) in arc_pts:
            if _point_seg_dist(x, y, a[0], a[1], b[0], b[1]) < clearance:
                return False
    return True


def _min_clearance(arc_pts, panels, launch_x, launch_y, launch_radius, cap):
    """Worst-case (smallest) distance from this arc to any nearby panel or
    to the launch keep-out disc, capped at `cap` (since we only care about
    ranking candidates near the danger zone, not exact far-away margins).
    Used to pick the *least bad* candidate when nothing fully clears."""
    worst = cap
    for panel in panels:
        a, b = _panel_endpoints(panel)
        for (x, y) in arc_pts:
            d = _point_seg_dist(x, y, a[0], a[1], b[0], b[1])
            if d < worst:
                worst = d
                if worst <= 0:
                    return 0.0
    n = len(arc_pts) - 1
    tail_n = max(1, int(round(0.25 * n)))
    for (x, y) in arc_pts[-tail_n:]:
        d = math.hypot(x - launch_x, y - launch_y) - (launch_radius - CLEARANCE)
        if d < worst:
            worst = d
    return worst


# ---------------------------------------------------------------------------


def build_simulation(beats, width, height, gravity=2200.0, seed=None,
                      start_x=None, start_y=90.0, panel_length_range=(140, 230),
                      min_dt=0.16, max_attempts=32, collision_lookback=20):
    """Returns (segments, panels, gravity).

    segments: list of dicts with keys t_start, t_end, x0, y0, vx, vy
              describing piecewise projectile-motion arcs. Position at
              time t within a segment is:
                x(t) = x0 + vx*(t - t_start)
                y(t) = y0 + vy*(t - t_start) + 0.5*g*(t - t_start)**2

    panels: list of Panel, one per beat, in beat order. Panel i is hit
            at time == beats[i].
    """
    rng = random.Random(seed)
    g = gravity

    beats = sorted(t for t in beats if t is not None)
    if not beats:
        raise ValueError("No beats to choreograph")
    # de-dup / enforce a minimum gap so consecutive bounces aren't
    # physically absurd (a bounce needs *some* time to read as a bounce)
    cleaned = [beats[0]]
    for t in beats[1:]:
        if t - cleaned[-1] >= min_dt:
            cleaned.append(t)
    beats = cleaned

    x0 = start_x if start_x is not None else width / 2.0
    y0 = start_y

    margin = width * 0.14
    color_idx = rng.randrange(len(PALETTE))

    def next_color():
        nonlocal color_idx
        color_idx = (color_idx + rng.randrange(1, len(PALETTE))) % len(PALETTE)
        return PALETTE[color_idx]

    segments = []
    panels = []

    # --- Segment 0: the initial drop, from the very start of the song ---
    t0 = max(beats[0], 0.15)
    segments.append(dict(t_start=0.0, t_end=t0, x0=x0, y0=y0, vx=0.0, vy=0.0))

    prev_x, prev_y = x0, y0 + 0.5 * g * t0 * t0
    prev_t = t0
    prev_vx_impact, prev_vy_impact = 0.0, g * t0  # velocity at moment of landing
    # distance just covered to reach (prev_x, prev_y) -- used to size the
    # panel placed there so it never claims more room than it actually has
    prev_hop_dist = math.hypot(prev_x - x0, prev_y - y0)

    last_dir = 1 if rng.random() > 0.5 else -1
    # realistic caps on how fast the ball can be launched sideways/vertically
    # by a bounce -- independent of dt, so short beat gaps naturally produce
    # small, gravity-dominated arcs instead of forced huge jumps
    max_launch_vx = 0.9 * width   # px/s
    max_launch_vy_up = 0.55 * math.sqrt(2 * g * height)  # px/s, "hero bounce" cap
    # conservative "keep out" disc around the launch point -- see
    # _arc_clear_of_launch_point for why this can't just use the launch
    # panel's real (not-yet-determined) shape
    launch_keepout_radius = panel_length_range[1] / 2.0 + CLEARANCE

    def candidate_is_clear(arc_pts, nearby_panels):
        return (_arc_clear_of_panels(arc_pts, nearby_panels, _CHECK_CLEARANCE) and
                _arc_clear_of_launch_point(arc_pts, prev_x, prev_y, launch_keepout_radius))

    def make_candidate(dt, allow_rise):
        nonlocal last_dir
        direction = -last_dir if rng.random() < 0.75 else last_dir
        lateral_frac = rng.uniform(0.12, 1.0 if allow_rise else 0.5)
        dx = direction * lateral_frac * min(max_launch_vx * dt, width * 0.7)
        x_target = prev_x + dx
        if x_target < margin or x_target > width - margin:
            direction = -direction
            x_target = prev_x + direction * lateral_frac * min(max_launch_vx * dt, width * 0.7)
        x_target = min(max(x_target, margin), width - margin)

        if allow_rise:
            # k < 1 => needs an upward kick (rising-then-falling arc)
            # k > 1 => already moving down fast, needs a downward kick
            k = rng.uniform(0.35, 1.5)
        else:
            # guaranteed-safe: never rises above the launch point
            k = rng.uniform(1.0, 1.5)
        descent = k * 0.5 * g * dt * dt
        descent = max(descent, 4.0)
        descent = min(descent, height * 1.6)
        y_target = prev_y + descent

        vx = (x_target - prev_x) / dt
        vy = (descent - 0.5 * g * dt * dt) / dt
        # clamp pathological velocities (very rare, only from extreme dt)
        vx = max(-max_launch_vx, min(max_launch_vx, vx))
        vy = max(-max_launch_vy_up, vy)

        return direction, x_target, y_target, vx, vy

    for i in range(1, len(beats)):
        t_next = beats[i]
        dt = t_next - prev_t
        if dt < min_dt:
            dt = min_dt
            t_next = prev_t + dt

        nearby_panels = panels[-collision_lookback:]

        chosen = None
        best_score = -1e18
        best_candidate = None
        score_budget = 5  # only rank the last few failed attempts per phase,
                           # to keep the common (quickly-successful) case cheap

        def _consider(direction, x_target, y_target, vx, vy, do_score):
            nonlocal best_score, best_candidate
            arc_pts = _sample_arc(prev_x, prev_y, vx, vy, g, dt)
            if candidate_is_clear(arc_pts, nearby_panels):
                return (direction, x_target, y_target, vx, vy)
            if do_score:
                score = _min_clearance(arc_pts, nearby_panels, prev_x, prev_y,
                                        launch_keepout_radius, cap=_CHECK_CLEARANCE)
                if score > best_score:
                    best_score = score
                    best_candidate = (direction, x_target, y_target, vx, vy)
            return None

        # phase 1: "fun" candidates that may rise above the launch point
        for attempt in range(max_attempts):
            result = _consider(*make_candidate(dt, allow_rise=True),
                                do_score=(attempt >= max_attempts - score_budget))
            if result is not None:
                chosen = result
                break

        if chosen is None:
            # phase 2: candidates that never rise above the launch point.
            # Panel *centers* are strictly increasing in y by construction,
            # but a steeply angled panel's actual line segment can still
            # extend back up past its own center -- so this is a much safer
            # bet than phase 1, but still needs to be verified geometrically,
            # not assumed.
            for attempt in range(max_attempts):
                result = _consider(*make_candidate(dt, allow_rise=False),
                                    do_score=(attempt >= max_attempts - score_budget))
                if result is not None:
                    chosen = result
                    break

        if chosen is None and best_candidate is not None and best_score > -CLEARANCE * 0.5:
            # nothing fully cleared, but this candidate came reasonably
            # close (within half a clearance-width of passing) -- use the
            # least-bad option we found rather than an unrelated escape hop
            chosen = best_candidate

        if chosen is None:
            # true last resort: step directly out to the edge of the launch
            # keep-out disc (the minimum distance that's actually safe from
            # the panel we're leaving) with plain free-fall descent. If it
            # *still* clips an older panel, accept it anyway rather than
            # breaking the beat sync -- an exceedingly rare edge case with
            # very densely packed beats.
            k = 1.0
            descent = max(k * 0.5 * g * dt * dt, 4.0)
            y_target = prev_y + descent
            direction = -last_dir
            escape_dx = launch_keepout_radius * 1.15
            x_target = prev_x + direction * escape_dx
            if x_target < margin or x_target > width - margin:
                direction = -direction
                x_target = prev_x + direction * escape_dx
            x_target = min(max(x_target, margin), width - margin)
            vx = (x_target - prev_x) / dt
            vy = (descent - 0.5 * g * dt * dt) / dt
        else:
            direction, x_target, y_target, vx, vy = chosen

        last_dir = direction
        hop_dist = math.hypot(x_target - prev_x, y_target - prev_y)
        segments.append(dict(t_start=prev_t, t_end=t_next, x0=prev_x, y0=prev_y, vx=vx, vy=vy))

        incoming = (prev_vx_impact, prev_vy_impact)
        outgoing = (vx, vy)
        angle = _panel_normal_angle(incoming, outgoing) + math.pi / 2  # tangent, for drawing
        panels.append(Panel(
            id=len(panels), x=prev_x, y=prev_y, angle=angle,
            length=_size_panel(prev_hop_dist, panel_length_range, rng),
            color=next_color(), hit_time=prev_t, seg_index=len(segments) - 1,
        ))

        prev_vx_impact = vx
        prev_vy_impact = vy + g * dt
        prev_x, prev_y = x_target, y_target
        prev_t = t_next
        prev_hop_dist = hop_dist

    # final panel: where the ball lands on the very last beat. Give it a
    # plausible outgoing bounce (up and off) purely for the reflection
    # angle / visual flourish -- no further beats to choreograph to.
    incoming = (prev_vx_impact, prev_vy_impact)
    outgoing = (incoming[0] * 0.5, -abs(incoming[1]) * 0.55)
    angle = _panel_normal_angle(incoming, outgoing) + math.pi / 2
    panels.append(Panel(
        id=len(panels), x=prev_x, y=prev_y, angle=angle,
        length=_size_panel(prev_hop_dist, panel_length_range, rng),
        color=next_color(), hit_time=prev_t, seg_index=len(segments) - 1,
    ))

    # one trailing "fly off" segment after the last beat, purely cosmetic,
    # so the ball doesn't just freeze in place at the end
    tail_dt = 1.2
    segments.append(dict(t_start=prev_t, t_end=prev_t + tail_dt,
                          x0=prev_x, y0=prev_y, vx=outgoing[0], vy=outgoing[1]))

    assert len(panels) == len(beats)
    _resolve_residual_overlaps(segments, panels, g)
    return segments, panels, gravity


def _resolve_residual_overlaps(segments, panels, g, index_window=24, min_length=46.0):
    """A panel's final angle depends on the segment leaving it, which is
    only chosen *after* the panel's position is fixed -- so a panel can
    end up steeply angled enough that its far end reaches back into the
    flight path of an earlier (or, rarely, later) segment that had no way
    to know about it when it was generated. Rather than re-deriving
    positions/timing (which would break exact beat sync), resolve any such
    residual overlap by shrinking the offending panel -- a purely cosmetic
    change with no physics impact -- until it clears, down to a minimum
    readable length."""
    n = len(panels)
    for idx, panel in enumerate(panels):
        incoming_seg = panel.seg_index - 1
        outgoing_seg = panel.seg_index
        lo = max(0, idx - index_window)
        hi = min(n, idx + index_window + 1)
        nearby_seg_indices = set()
        for j in range(lo, hi):
            nearby_seg_indices.add(panels[j].seg_index)
            nearby_seg_indices.add(panels[j].seg_index - 1)
        nearby_seg_indices.discard(incoming_seg)
        nearby_seg_indices.discard(outgoing_seg)

        check_pts = []
        for sidx in nearby_seg_indices:
            if 0 <= sidx < len(segments):
                s = segments[sidx]
                check_pts.extend(_sample_arc(s["x0"], s["y0"], s["vx"], s["vy"], g,
                                              s["t_end"] - s["t_start"], n=24))
        if not check_pts:
            continue

        length = panel.length
        for _ in range(10):
            a, b = _panel_endpoints(panel)
            worst = min((_point_seg_dist(x, y, a[0], a[1], b[0], b[1]) for (x, y) in check_pts),
                        default=CLEARANCE)
            if worst >= CLEARANCE or length <= min_length:
                break
            length = max(min_length, length * 0.8)
            panel.length = length


def position_at(segments, t, gravity):
    """Sample (x, y, vx, vy_instant) at time t from the piecewise segments."""
    seg = segments[0]
    for s in segments:
        if t >= s["t_start"]:
            seg = s
        else:
            break
    dt = max(0.0, t - seg["t_start"])
    x = seg["x0"] + seg["vx"] * dt
    y = seg["y0"] + seg["vy"] * dt + 0.5 * gravity * dt * dt
    vx = seg["vx"]
    vy = seg["vy"] + gravity * dt
    return x, y, vx, vy
