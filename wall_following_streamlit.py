"""
Streamlit Dashboard: Warehouse AGV MDP Navigation (Pre-drawn Path & Clean Layout)
=================================================================================
Run with:
    streamlit run wall_following_streamlit.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Warehouse AGV MDP Navigation", layout="wide")

# ---------------------------------------------------------------------
# 1. Load Dataset & MDP Policies
# ---------------------------------------------------------------------
@st.cache_data
def load_resources():
    df = pd.read_csv('sensor_readings_4.csv', header=None, names=['SD_front', 'SD_left', 'SD_right', 'SD_back', 'Class'])
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

# ---------------------------------------------------------------------
# 2. Pre-compute Exact Trajectory & Dynamic Gate Coordinates
# ---------------------------------------------------------------------
def precompute_trajectory(actions):
    """
    Generates the full rectangular circuit matching the dataset path.
    """
    traj = []
    x_min, x_max = 1.0, 11.0
    y_min, y_max = 1.0, 9.0
    
    # Start at entry point
    x, y = 2.0, 9.0
    heading = 0  # 0: East, 1: South, 2: West, 3: North
    
    dx_list = [1.0, 0.0, -1.0, 0.0]
    dy_list = [0.0, -1.0, 0.0, 1.0]
    
    for action in actions:
        if action == 'Sharp-Right-Turn':
            heading = (heading + 1) % 4
            step = 0.03
        elif action == 'Slight-Right-Turn':
            step = 0.04
        elif action == 'Slight-Left-Turn':
            heading = (heading - 1) % 4
            step = 0.03
        else: # Move-Forward
            step = 0.05
            
        x += dx_list[heading] * step
        y += dy_list[heading] * step
        
        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)
        
        traj.append({'x': x, 'y': y})
    return traj

PRECOMPUTED_TRAJ = precompute_trajectory(DATA['Class'].values)

ENTRY_POS = PRECOMPUTED_TRAJ[0]
EXIT_POS = PRECOMPUTED_TRAJ[-1]
max_steps = len(DATA)

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
    ss = st.session_state
    if ss.idx >= max_steps:
        ss.finished = True
        ss.running = False
        return

    row = DATA.iloc[ss.idx]
    sd_left = row['SD_left']
    dataset_action = row['Class']

    # Optimal classification thresholds hardcoded
    if sd_left < 0.53:
        state = 'Too-Close'
    elif sd_left > 0.71:
        state = 'Too-Far'
    else:
        state = 'Ideal'

    action = POLICY.get(state, dataset_action)
    if action not in R[state]:
        action = 'Move-Forward'

    reward = R[state][action]
    ss.cum_reward += reward

    ss.log.append(f"Step {ss.idx:04d} | State: {state:<10s} | Action: {action:<18s}")
    if len(ss.log) > 100:
        ss.log.pop(0)

    ss.idx += 1

# ---------------------------------------------------------------------
# 4. Streamlit Dashboard Layout (Larger Motion Window & Compact Legend)
# ---------------------------------------------------------------------
st.title("📦 Warehouse AGV MDP Optimal Navigation")
st.markdown("AGV cursor tracing the **pre-drawn optimal dataset path** from the Entry Gate around storage racks to the Exit Gate.")

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

    current_row = DATA.iloc[min(st.session_state.idx, len(DATA)-1)]
    cur_state = 'Too-Close' if current_row['SD_left'] < 0.53 else ('Too-Far' if current_row['SD_left'] > 0.71 else 'Ideal')
    st.info(f"**Current State:** `{cur_state}`\n\n"
            f"• Left Distance: `{current_row['SD_left']:.2f}m`\n"
            f"• Front Distance: `{current_row['SD_front']:.2f}m`")

    st.subheader("📜 Event Log")
    st.code("\n".join(st.session_state.log[-10:]) if st.session_state.log else "—", language=None)

# ---------------------------------------------------------------------
# 5. Warehouse Blueprint & Pre-drawn Path Rendering
# ---------------------------------------------------------------------
def render_warehouse():
    ss = st.session_state
    # Increased plot size for a larger motion window
    fig, ax = plt.subplots(figsize=(9, 7.5))
    
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.set_aspect('equal')

    # Warehouse Outer Walls
    ax.plot([0, 12, 12, 0, 0], [0, 0, 10, 10, 0], color='#38bdf8', linewidth=3.5, label="Walls")

    # Entry and Exit Gates
    ax.scatter([ENTRY_POS['x']], [ENTRY_POS['y']], color='#22c55e', s=200, zorder=6, marker='o', edgecolors='white', linewidths=1.5, label="Entry")
    ax.scatter([EXIT_POS['x']], [EXIT_POS['y']], color='#ef4444', s=200, zorder=6, marker='X', edgecolors='white', linewidths=1.5, label="Exit")

    # Internal Fulfillment Storage Racks (Obstacles)
    racks = [
        ([2.5, 5.0, 5.0, 2.5, 2.5], [2.0, 2.0, 4.5, 4.5, 2.0]),
        ([7.0, 9.5, 9.5, 7.0, 7.0], [2.0, 2.0, 4.5, 4.5, 2.0]),
        ([2.5, 5.0, 5.0, 2.5, 2.5], [5.5, 5.5, 8.0, 8.0, 5.5]),
        ([7.0, 9.5, 9.5, 7.0, 7.0], [5.5, 5.5, 8.0, 8.0, 5.5]),
    ]
    for rx, ry in racks:
        ax.fill(rx, ry, color='#1e293b', edgecolor='#64748b', linewidth=1.5, zorder=2)
        ax.text(np.mean(rx), np.mean(ry), "RACK", color='#64748b', fontsize=7, ha='center', va='center', fontweight='bold', alpha=0.7, zorder=3)

    # 1. PRE-DRAWN FULL PATH (Entire path visible in background)
    all_tx = [p['x'] for p in PRECOMPUTED_TRAJ]
    all_ty = [p['y'] for p in PRECOMPUTED_TRAJ]
    ax.plot(all_tx, all_ty, color='#1e3a8a', linewidth=2.0, linestyle='--', label="Full Path", zorder=4)

    # 2. TRAVELLED PATH TRACE (Bright trace showing progress up to current step)
    current_idx = min(ss.idx, len(PRECOMPUTED_TRAJ) - 1)
    if current_idx > 0:
        travelled_traj = PRECOMPUTED_TRAJ[: current_idx + 1]
        ttx = [p['x'] for p in travelled_traj]
        tty = [p['y'] for p in travelled_traj]
        ax.plot(ttx, tty, color='#38bdf8', linewidth=3.0, alpha=0.9, label="Traveled", zorder=5)

    # 3. AGV Cursor Head
    cur_pos = PRECOMPUTED_TRAJ[current_idx]
    ax.scatter([cur_pos['x']], [cur_pos['y']], color='#facc15', s=220, zorder=7, marker='s', edgecolors='white', linewidths=1.5, label="AGV")

    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 11)
    
    # Compact Legend placed cleanly in the corner
    ax.legend(loc='upper right', framealpha=0.6, fontsize=6.5, facecolor='#1e293b', edgecolor='none', labelcolor='white', ncol=2)
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
