"""
Streamlit Dashboard: Warehouse AGV MDP Navigation (Logical Obstacle Walls & Corridors)
=====================================================================================
Run with:
    streamlit run wall_following_streamlit.py

Changes vs. the previous version
---------------------------------
1. The path drawn on the map and the path actually simulated are now the SAME
   trajectory. Previously the background path was traced from the dataset's raw
   `Class` column while the live simulation independently looked up actions from
   the optimal-policy CSV -- two unrelated action streams sharing one plot. Now
   there is a single precomputed sequence: state -> policy lookup -> action ->
   position, computed once, used both for the drawing and for playback.
2. Storage racks are no longer a handful of hardcoded rectangles. They're
   inferred from the trajectory itself: every grid cell the AGV never visits
   is rendered as a rack, so the "warehouse" is shaped by the actual data.
3. The arena is sized to the trajectory's real bounding box (with padding),
   instead of a fixed 12x10 box. The previous fixed box clipped ~17% of steps
   against its walls, which distorted the shape of the corridor.

Section 6 (added)
------------------
Below the original live-playback dashboard, a second section compares four
extra ways of deriving a policy over the same 3-state x 4-action problem,
rendered on the SAME warehouse layout (same walls/racks, same entry/exit) so
the resulting paths are directly comparable:

    - Approximate Dynamic Programming (approximate policy iteration over an
      empirically-estimated transition model P(s'|s))
    - Monte Carlo control (first-visit, epsilon-greedy, sampled episodes)
    - Local search (hill-climbing with random restarts over the policy table)
    - Hooke-Jeeves pattern search (exploratory + pattern moves over an
      integer action-index encoding of the policy)

Nothing in Sections 1-5 below was modified -- Section 6 only reads the
objects they already define (DATA, POLICY, R, classify_state, BOUNDS,
RACK_CELLS, RACK_CELL_SIZE, ENTRY_POS, EXIT_POS, ENTRY_WALL, EXIT_WALL,
draw_wall_with_gap).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Warehouse AGV MDP Navigation", layout="wide")

# ---------------------------------------------------------------------
# 1. Load Dataset & MDP Policy
# ---------------------------------------------------------------------
@st.cache_data
def load_resources():
    df = pd.read_csv('sensor_readings_4.csv', header=None,
                      names=['SD_front', 'SD_left', 'SD_right', 'SD_back', 'Class'])
    try:
        opt_df = pd.read_csv('optimal_value_function.csv')
        policy = dict(zip(opt_df['State'], opt_df['Optimal_Action']))
        values = dict(zip(opt_df['State'], opt_df['Optimal_Value']))
    except Exception:
        policy = {'Too-Close': 'Sharp-Right-Turn', 'Ideal': 'Move-Forward', 'Too-Far': 'Slight-Left-Turn'}
        values = {'Too-Close': 100.0, 'Ideal': 100.0, 'Too-Far': 100.0}
    return df, policy, values

DATA, POLICY, VALUES = load_resources()

R = {
    'Too-Close': {'Move-Forward': -10, 'Slight-Right-Turn': 5, 'Sharp-Right-Turn': 10, 'Slight-Left-Turn': -10},
    'Ideal':     {'Move-Forward': 10, 'Slight-Right-Turn': 2, 'Sharp-Right-Turn': -5, 'Slight-Left-Turn': 2},
    'Too-Far':   {'Move-Forward': -2, 'Slight-Right-Turn': -10, 'Sharp-Right-Turn': -10, 'Slight-Left-Turn': 10},
}


def classify_state(sd_left: float) -> str:
    """Single source of truth for turning a left-distance reading into an MDP state."""
    if sd_left < 0.53:
        return 'Too-Close'
    elif sd_left > 0.71:
        return 'Too-Far'
    else:
        return 'Ideal'


# ---------------------------------------------------------------------
# 2. Precompute the ONE trajectory the policy actually produces
#    (state -> policy -> action -> position), used for both the
#    drawn path and the live simulation. Also derive the arena size
#    from where this trajectory actually goes.
# ---------------------------------------------------------------------
@st.cache_data
def precompute_run(_data: pd.DataFrame, _policy: dict):
    dx_list = [1.0, 0.0, -1.0, 0.0]   # heading: 0=E, 1=S, 2=W, 3=N
    dy_list = [0.0, -1.0, 0.0, 1.0]

    x, y, heading = 0.0, 0.0, 0
    states, actions, rewards, xs, ys = [], [], [], [], []
    cum = 0.0
    cum_rewards = []

    for sd_left in _data['SD_left'].values:
        state = classify_state(sd_left)
        action = _policy.get(state, 'Move-Forward')
        if action not in R[state]:
            action = 'Move-Forward'

        if action == 'Sharp-Right-Turn':
            heading = (heading + 1) % 4
            step = 0.03
        elif action == 'Slight-Right-Turn':
            step = 0.04
        elif action == 'Slight-Left-Turn':
            heading = (heading - 1) % 4
            step = 0.03
        else:
            step = 0.05

        x += dx_list[heading] * step
        y += dy_list[heading] * step

        reward = R[state][action]
        cum += reward

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        cum_rewards.append(cum)
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    # Size the arena to where the path actually goes, with padding on every side
    # so the AGV never has to be clamped against a wall.
    pad = 1.0
    x_min, x_max = xs.min() - pad, xs.max() + pad
    y_min, y_max = ys.min() - pad, ys.max() + pad

    return {
        'xs': xs, 'ys': ys,
        'states': states, 'actions': actions,
        'rewards': rewards, 'cum_rewards': cum_rewards,
        'bounds': (x_min, x_max, y_min, y_max),
    }


@st.cache_data
def build_obstacle_grid(xs, ys, bounds, cell=0.5, pad_cells=1):
    """Rasterize the trajectory: any cell the path never enters (plus a small
    buffer so the corridor has walking room) becomes a storage rack."""
    x_min, x_max, y_min, y_max = bounds
    nx = max(1, int(np.ceil((x_max - x_min) / cell)))
    ny = max(1, int(np.ceil((y_max - y_min) / cell)))

    corridor = np.zeros((ny, nx), dtype=bool)
    for x, y in zip(xs, ys):
        ci = int((x - x_min) / cell)
        cj = int((y - y_min) / cell)
        for di in range(-pad_cells, pad_cells + 1):
            for dj in range(-pad_cells, pad_cells + 1):
                ii, jj = ci + di, cj + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    corridor[jj, ii] = True

    rack_cells = [(x_min + i * cell, y_min + j * cell)
                  for j in range(ny) for i in range(nx) if not corridor[j, i]]
    return rack_cells, cell


RUN = precompute_run(DATA, POLICY)
XS, YS = RUN['xs'], RUN['ys']
BOUNDS = RUN['bounds']
RACK_CELLS, RACK_CELL_SIZE = build_obstacle_grid(XS, YS, BOUNDS)

ENTRY_POS = (XS[0], YS[0])
EXIT_POS = (XS[-1], YS[-1])
max_steps = len(DATA)


def snap_to_boundary(pt, bounds):
    x, y = pt
    x_min, x_max, y_min, y_max = bounds
    d = {'left': abs(x - x_min), 'right': abs(x - x_max),
         'bottom': abs(y - y_min), 'top': abs(y - y_max)}
    side = min(d, key=d.get)
    if side == 'left':
        return (x_min, y)
    if side == 'right':
        return (x_max, y)
    if side == 'bottom':
        return (x, y_min)
    return (x, y_max)


ENTRY_WALL = snap_to_boundary(ENTRY_POS, BOUNDS)
EXIT_WALL = snap_to_boundary(EXIT_POS, BOUNDS)

# ---------------------------------------------------------------------
# 3. Session State Initialization
# ---------------------------------------------------------------------
def reset_sim():
    st.session_state.idx = 0
    st.session_state.cum_reward = 0
    st.session_state.log = []
    st.session_state.running = False
    st.session_state.finished = False

if 'idx' not in st.session_state:
    reset_sim()

def step_simulation():
    """Advance one step using the precomputed run -- no re-deriving state/
    action here, so the displayed path and the simulated path can never
    drift apart."""
    ss = st.session_state
    if ss.idx >= max_steps:
        ss.finished = True
        ss.running = False
        return

    state = RUN['states'][ss.idx]
    action = RUN['actions'][ss.idx]
    ss.cum_reward = RUN['cum_rewards'][ss.idx]

    ss.log.append(f"Step {ss.idx:04d} | State: {state:<10s} | Action: {action:<18s}")
    if len(ss.log) > 100:
        ss.log.pop(0)

    ss.idx += 1

# ---------------------------------------------------------------------
# 4. Streamlit Dashboard Layout
# ---------------------------------------------------------------------
st.title("📦 Warehouse AGV MDP Optimal Navigation")
st.markdown(
    "AGV cursor following the **policy-driven** path from Entry to Exit. "
    "Storage racks are inferred from cells the path never visits; the arena "
    "is sized to fit the path so it never clips against a wall."
)

col_left, col_right = st.columns([3.0, 1.0])

with col_left:
    btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 2])
    if btn1.button("▶ Play / Pause"):
        st.session_state.running = not st.session_state.running
    if btn2.button("Step ⏭"):
        step_simulation()
    if btn3.button("↺ Reset"):
        reset_sim()
    sim_speed = btn4.slider("Playback Speed", 0.01, 0.2, 0.02, 0.01)

    plot_placeholder = st.empty()

with col_right:
    st.subheader("📊 AGV Status")
    m1, m2 = st.columns(2)
    m1.metric("Progress", f"{st.session_state.idx} / {max_steps}")
    m2.metric("Reward", st.session_state.cum_reward)

    cur_idx = min(st.session_state.idx, max_steps - 1)
    cur_state = RUN['states'][cur_idx]
    cur_action = RUN['actions'][cur_idx]
    current_row = DATA.iloc[cur_idx]
    st.info(f"**Current State:** `{cur_state}`  →  **Action:** `{cur_action}`\n\n"
            f"• Left Distance: `{current_row['SD_left']:.2f}m`\n"
            f"• Front Distance: `{current_row['SD_front']:.2f}m`")

    st.subheader("📜 Event Log")
    st.code("\n".join(st.session_state.log[-10:]) if st.session_state.log else "—", language=None)

# ---------------------------------------------------------------------
# 5. Warehouse Blueprint & Obstacle Corridor Rendering
# ---------------------------------------------------------------------
def draw_wall_with_gap(ax, bounds, gap_center, gap_width=1.0, color='#38bdf8'):
    x_min, x_max, y_min, y_max = bounds
    corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
    gx, gy = gap_center
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        n = 200
        xs_seg = np.linspace(x0, x1, n)
        ys_seg = np.linspace(y0, y1, n)
        d = np.hypot(xs_seg - gx, ys_seg - gy)
        mask = d > gap_width / 2
        seg_x, seg_y = [], []
        for k in range(n):
            if mask[k]:
                seg_x.append(xs_seg[k]); seg_y.append(ys_seg[k])
            else:
                if seg_x:
                    ax.plot(seg_x, seg_y, color=color, linewidth=4, zorder=6, solid_capstyle='butt')
                seg_x, seg_y = [], []
        if seg_x:
            ax.plot(seg_x, seg_y, color=color, linewidth=4, zorder=6, solid_capstyle='butt')


def render_warehouse():
    ss = st.session_state
    x_min, x_max, y_min, y_max = BOUNDS
    fig, ax = plt.subplots(figsize=(9, 7.5))

    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.set_aspect('equal')

    # Storage racks inferred from unvisited grid cells
    for rx, ry in RACK_CELLS:
        ax.add_patch(Rectangle((rx, ry), RACK_CELL_SIZE, RACK_CELL_SIZE,
                                facecolor='#1e293b', edgecolor='#334155',
                                linewidth=0.4, zorder=2))

    # Outer boundary wall, with a gap at the entry and a gap at the exit
    draw_wall_with_gap(ax, BOUNDS, ENTRY_WALL, gap_width=1.0, color='#38bdf8')
    draw_wall_with_gap(ax, BOUNDS, EXIT_WALL, gap_width=1.0, color='#38bdf8')

    # Entry and Exit markers
    ax.scatter([ENTRY_POS[0]], [ENTRY_POS[1]], color='#22c55e', s=200, zorder=7,
               marker='o', edgecolors='white', linewidths=1.5, label="Entry")
    ax.scatter([EXIT_POS[0]], [EXIT_POS[1]], color='#ef4444', s=200, zorder=7,
               marker='X', edgecolors='white', linewidths=1.5, label="Exit")

    # 1. Full target path (the same trajectory the simulation plays back)
    ax.plot(XS, YS, color='#1d4ed8', linewidth=1.6, linestyle='--', alpha=0.7,
            label="Target Path", zorder=5)

    # 2. Traveled path trace
    current_idx = min(ss.idx, len(XS) - 1)
    if current_idx > 0:
        ax.plot(XS[:current_idx + 1], YS[:current_idx + 1], color='#38bdf8',
                linewidth=3.0, alpha=0.9, label="Traveled", zorder=6)

    # 3. AGV cursor head
    ax.scatter([XS[current_idx]], [YS[current_idx]], color='#facc15', s=220, zorder=8,
               marker='s', edgecolors='white', linewidths=1.5, label="AGV")

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    ax.legend(loc='upper right', framealpha=0.6, fontsize=6.5, facecolor='#1e293b',
              edgecolor='none', labelcolor='white', ncol=2)
    ax.axis('off')

    plot_placeholder.pyplot(fig, use_container_width=True)
    plt.close(fig)

render_warehouse()

if st.session_state.running and not st.session_state.finished:
    for _ in range(15):
        if not st.session_state.running or st.session_state.finished:
            break
        step_simulation()
        render_warehouse()
        time.sleep(sim_speed)
    st.rerun()

# =======================================================================
# 6. NEW: Comparing alternative policy-search methods (same layout)
# =======================================================================
# Everything below is additive. It reuses R, classify_state, DATA, BOUNDS,
# RACK_CELLS, RACK_CELL_SIZE, ENTRY_POS/EXIT_POS, ENTRY_WALL/EXIT_WALL and
# draw_wall_with_gap from the sections above without modifying any of them.
#
# Framing: the "policy" in this problem is just a lookup table with three
# entries (one action per state), and there are only 4**3 = 64 possible
# deterministic policies. That's small enough to treat literally as a
# policy-search problem -- each method below searches (exactly, empirically,
# or heuristically) over that table, using the same reward matrix R.
#
# Honesty note shown in the UI too: in this dataset the *next* state depends
# only on the next sensor reading, not on the action taken, so the total
# reward of a policy is separable per state and value iteration already
# finds the global optimum (that's the "Optimal_Action" column you loaded).
# Approximate DP will therefore tend to rediscover the same optimum; Monte
# Carlo and the two heuristic search methods are deliberately kept
# sample/iteration-limited so their tabs show real, sometimes-imperfect
# approximation behavior instead of trivially matching it.

STATES = ['Too-Close', 'Ideal', 'Too-Far']
ACTIONS = ['Move-Forward', 'Slight-Right-Turn', 'Sharp-Right-Turn', 'Slight-Left-Turn']


@st.cache_data
def build_state_sequence():
    return [classify_state(v) for v in DATA['SD_left'].values]


@st.cache_data
def build_state_stats():
    seq = build_state_sequence()
    counts = {s: 0 for s in STATES}
    trans = {s: {s2: 0 for s2 in STATES} for s in STATES}
    for i, s in enumerate(seq):
        counts[s] += 1
        if i + 1 < len(seq):
            trans[s][seq[i + 1]] += 1
    P = {}
    for s in STATES:
        total = sum(trans[s].values())
        if total == 0:
            P[s] = {s2: 1.0 / len(STATES) for s2 in STATES}
        else:
            P[s] = {s2: trans[s][s2] / total for s2 in STATES}
    return counts, P


STATE_SEQ = build_state_sequence()
STATE_COUNTS, TRANSITION_P = build_state_stats()


def policy_fitness(policy):
    """Exact total reward a policy earns over the whole recorded run.
    (Exact, not simulated with noise, because state transitions here don't
    depend on the action taken -- see the honesty note above.)"""
    return sum(STATE_COUNTS[s] * R[s][policy.get(s, 'Move-Forward')] for s in STATES)


def simulate_positions(policy):
    """Same step rule as precompute_run's inner loop, but for an arbitrary
    policy and without touching that cached function or its bounds -- we
    deliberately plot every method inside the SAME BOUNDS/RACK_CELLS so the
    room layout is identical across tabs."""
    dx_list = [1.0, 0.0, -1.0, 0.0]
    dy_list = [0.0, -1.0, 0.0, 1.0]
    x, y, heading = 0.0, 0.0, 0
    xs, ys, cum_rewards = [], [], []
    cum = 0.0
    for state in STATE_SEQ:
        action = policy.get(state, 'Move-Forward')
        if action not in R[state]:
            action = 'Move-Forward'
        if action == 'Sharp-Right-Turn':
            heading = (heading + 1) % 4
            step = 0.03
        elif action == 'Slight-Right-Turn':
            step = 0.04
        elif action == 'Slight-Left-Turn':
            heading = (heading - 1) % 4
            step = 0.03
        else:
            step = 0.05
        x += dx_list[heading] * step
        y += dy_list[heading] * step
        cum += R[state][action]
        xs.append(x)
        ys.append(y)
        cum_rewards.append(cum)
    return np.array(xs), np.array(ys), cum_rewards


# --- 6a. Approximate Dynamic Programming (approximate policy iteration) ---
def run_approx_dp(gamma=0.9, iterations=15, eval_sweeps=5, seed=0):
    rng = np.random.default_rng(seed)
    policy = {s: rng.choice(ACTIONS) for s in STATES}
    V = {s: 0.0 for s in STATES}
    history = [policy_fitness(policy)]
    for _ in range(iterations):
        # Approximate policy evaluation: a handful of bootstrapped Bellman
        # sweeps against the EMPIRICALLY ESTIMATED transition model, rather
        # than solving the linear system exactly -- this is the
        # "approximate" part of approximate policy iteration.
        for _ in range(eval_sweeps):
            V = {s: R[s][policy[s]] + gamma * sum(TRANSITION_P[s][s2] * V[s2] for s2 in STATES)
                 for s in STATES}
        # Policy improvement
        stable = True
        new_policy = {}
        for s in STATES:
            q = {a: R[s][a] + gamma * sum(TRANSITION_P[s][s2] * V[s2] for s2 in STATES) for a in ACTIONS}
            best_a = max(q, key=q.get)
            if best_a != policy[s]:
                stable = False
            new_policy[s] = best_a
        policy = new_policy
        history.append(policy_fitness(policy))
        if stable:
            break
    return policy, history


# --- 6b. Monte Carlo control (first-visit, epsilon-greedy) ---
def run_monte_carlo(episodes=200, episode_len=10, gamma=0.3, epsilon_start=0.5, seed=1):
    rng = np.random.default_rng(seed)
    Q = {s: {a: 0.0 for a in ACTIONS} for s in STATES}
    N = {s: {a: 0 for a in ACTIONS} for s in STATES}
    n = len(STATE_SEQ)
    history = []
    for ep in range(episodes):
        epsilon = max(0.05, epsilon_start * (1 - ep / episodes))
        start = int(rng.integers(0, max(1, n - episode_len)))
        seq = STATE_SEQ[start:start + episode_len]

        trajectory = []
        for s in seq:
            if rng.random() < epsilon:
                a = rng.choice(ACTIONS)
            else:
                a = max(Q[s], key=Q[s].get)
            trajectory.append((s, a, R[s][a]))

        # First-visit Monte Carlo return, backed up from the end of episode
        G = 0.0
        visited = set()
        for t in reversed(range(len(trajectory))):
            s, a, r = trajectory[t]
            G = r + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                N[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / N[s][a]

        greedy_policy = {s: max(Q[s], key=Q[s].get) for s in STATES}
        history.append(policy_fitness(greedy_policy))

    policy = {s: max(Q[s], key=Q[s].get) for s in STATES}
    return policy, history


# --- 6c. Local search / hill-climbing over the policy table ---
def run_local_search(iterations=25, restart_prob=0.10, seed=2):
    rng = np.random.default_rng(seed)
    current = {s: rng.choice(ACTIONS) for s in STATES}
    current_fit = policy_fitness(current)
    best, best_fit = dict(current), current_fit
    history = [best_fit]
    for _ in range(iterations):
        neighbor = dict(current)
        s = rng.choice(STATES)
        neighbor[s] = rng.choice(ACTIONS)
        neighbor_fit = policy_fitness(neighbor)
        if neighbor_fit >= current_fit or rng.random() < restart_prob:
            current, current_fit = neighbor, neighbor_fit
        if current_fit > best_fit:
            best, best_fit = dict(current), current_fit
        history.append(best_fit)
    return best, history


# --- 6d. Hooke-Jeeves pattern search (integer action-index encoding) ---
def run_hooke_jeeves(step_init=2, seed=3):
    rng = np.random.default_rng(seed)
    idx = {s: int(rng.integers(0, len(ACTIONS))) for s in STATES}

    def to_policy(vec):
        return {s: ACTIONS[max(0, min(len(ACTIONS) - 1, int(round(vec[s]))))] for s in STATES}

    def f(vec):
        return policy_fitness(to_policy(vec))

    def explore(base, step):
        point = dict(base)
        improved = False
        for s in STATES:
            best_val = f(point)
            best_point = point
            for delta in (step, -step):
                trial = dict(point)
                trial[s] = max(0, min(len(ACTIONS) - 1, point[s] + delta))
                if f(trial) > best_val:
                    best_val = f(trial)
                    best_point = trial
                    improved = True
            point = best_point
        return point, improved

    base = idx
    step = step_init
    history = [f(base)]
    while step >= 1:
        trial, improved = explore(base, step)
        if improved:
            # Pattern move: push further along the direction that just worked
            pattern = {s: max(0, min(len(ACTIONS) - 1, base[s] + 2 * (trial[s] - base[s]))) for s in STATES}
            pattern_explored, _ = explore(pattern, step)
            base = pattern_explored if f(pattern_explored) > f(trial) else trial
            history.append(f(base))
        else:
            step = step // 2  # shrink the step and search more finely
    return to_policy(base), history


def plot_policy_path(title, xs, ys, color):
    """Static path-on-layout plot for one method's tab. Uses the SAME
    BOUNDS / RACK_CELLS / entry / exit as the live dashboard above, so the
    room itself never changes between tabs -- only the path does."""
    x_min, x_max, y_min, y_max = BOUNDS
    fig, ax = plt.subplots(figsize=(6, 5.2))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.set_aspect('equal')

    for rx, ry in RACK_CELLS:
        ax.add_patch(Rectangle((rx, ry), RACK_CELL_SIZE, RACK_CELL_SIZE,
                                facecolor='#1e293b', edgecolor='#334155',
                                linewidth=0.4, zorder=2))

    draw_wall_with_gap(ax, BOUNDS, ENTRY_WALL, gap_width=1.0, color='#38bdf8')
    draw_wall_with_gap(ax, BOUNDS, EXIT_WALL, gap_width=1.0, color='#38bdf8')

    ax.scatter([ENTRY_POS[0]], [ENTRY_POS[1]], color='#22c55e', s=140, zorder=7,
               marker='o', edgecolors='white', linewidths=1.2)
    ax.scatter([EXIT_POS[0]], [EXIT_POS[1]], color='#ef4444', s=140, zorder=7,
               marker='X', edgecolors='white', linewidths=1.2)

    ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.9, zorder=6)
    ax.scatter([xs[-1]], [ys[-1]], color='#facc15', s=150, zorder=8,
               marker='s', edgecolors='white', linewidths=1.2)

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_title(title, color='white', fontsize=11)
    ax.axis('off')
    return fig


st.markdown("---")
st.header("🔬 Comparing Policy-Search Methods (Same Warehouse Layout)")
st.markdown(
    "Same room, same walls, same storage racks, same reward matrix `R` -- only "
    "*how the policy was found* changes between tabs. Because the next state "
    "in this dataset depends on the next sensor reading rather than on the "
    "action taken, the exact methods (value iteration, approximate DP) tend "
    "to converge on the same optimum already used above; Monte Carlo and the "
    "two heuristic searches are kept sample/iteration-limited so you can see "
    "them approach -- and sometimes miss -- that optimum."
)

with st.spinner("Running policy-search methods..."):
    adp_policy, adp_hist = run_approx_dp()
    mc_policy, mc_hist = run_monte_carlo()
    ls_policy, ls_hist = run_local_search()
    hj_policy, hj_hist = run_hooke_jeeves()

METHODS = {
    "Value Iteration (baseline)": {"policy": POLICY, "history": None, "color": "#38bdf8",
                                    "note": "Loaded directly from optimal_value_function.csv -- exact solution."},
    "Approximate DP": {"policy": adp_policy, "history": adp_hist, "color": "#a855f7",
                        "note": "Approximate policy iteration over an empirically estimated P(s'|s)."},
    "Monte Carlo": {"policy": mc_policy, "history": mc_hist, "color": "#f97316",
                     "note": "First-visit MC control, epsilon-greedy, 200 short sampled episodes. "
                             "Watch the fitness chart -- it starts noisy and settles as Q(s,a) converges."},
    "Local Search": {"policy": ls_policy, "history": ls_hist, "color": "#22c55e",
                      "note": "Hill-climbing over the policy table with random restarts."},
    "Hooke-Jeeves": {"policy": hj_policy, "history": hj_hist, "color": "#eab308",
                      "note": "Pattern search over an integer action-index encoding of the policy."},
}

tabs = st.tabs(list(METHODS.keys()))
summary = []

for tab, name in zip(tabs, METHODS.keys()):
    info = METHODS[name]
    policy = info["policy"]
    xs_m, ys_m, cum_rewards_m = simulate_positions(policy)
    total_reward = cum_rewards_m[-1]
    summary.append({"Method": name, "Total Reward": total_reward})

    with tab:
        c1, c2 = st.columns([2.0, 1.2])
        with c1:
            fig = plot_policy_path(name, xs_m, ys_m, info["color"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            st.caption(info["note"])
            st.metric("Total Reward (full run)", f"{total_reward:.0f}")
            st.table(pd.DataFrame(
                [{"State": s, "Action": policy.get(s, "Move-Forward")} for s in STATES]
            ).set_index("State"))
            if info["history"] is not None:
                st.caption("Fitness over the course of the search")
                st.line_chart(pd.DataFrame({"Total Reward": info["history"]}))

st.subheader("📊 Total Reward by Method")
summary_df = pd.DataFrame(summary).set_index("Method")
st.bar_chart(summary_df)
