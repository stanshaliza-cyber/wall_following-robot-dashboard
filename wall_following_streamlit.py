"""
Streamlit Dashboard: Warehouse AGV / Snake-Style MDP Navigation
==============================================================
Run with:
    streamlit run wall_following_streamlit.py
"""

import time
import math
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
# 2. Sidebar Controls
# ---------------------------------------------------------------------
st.sidebar.header("🏭 Warehouse Settings")
too_close_thresh = st.sidebar.slider("Too-Close Threshold (m)", 0.3, 0.6, 0.53, 0.01)
too_far_thresh = st.sidebar.slider("Too-Far Threshold (m)", 0.6, 1.2, 0.71, 0.01)
max_steps = st.sidebar.slider("Simulation Steps", 500, len(DATA), 2000, 100)
trail_length = st.sidebar.slider("Snake Trail Length", 20, 300, 100, 10)

def classify_state(sd_left):
    if sd_left < too_close_thresh:
        return 'Too-Close'
    elif sd_left > too_far_thresh:
        return 'Too-Far'
    else:
        return 'Ideal'

# ---------------------------------------------------------------------
# 3. Session State Initialization
# ---------------------------------------------------------------------
def reset_sim():
    st.session_state.idx = 0
    st.session_state.x = 1.5
    st.session_state.y = 8.5
    st.session_state.heading_idx = 0  # 0: East, 1: South, 2: West, 3: North
    st.session_state.cum_reward = 0
    st.session_state.trajectory = [{'x': 1.5, 'y': 8.5}]
    st.session_state.log = []
    st.session_state.running = False
    st.session_state.finished = False

if 'idx' not in st.session_state:
    reset_sim()

DIRECTIONS = [(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)]

def step_simulation():
    ss = st.session_state
    if ss.idx >= max_steps:
        ss.finished = True
        ss.running = False
        return

    row = DATA.iloc[ss.idx]
    sd_left = row['SD_left']
    dataset_action = row['Class']

    state = classify_state(sd_left)
    action = POLICY.get(state, dataset_action)
    if action not in R[state]:
        action = 'Move-Forward'

    reward = R[state][action]
    ss.cum_reward += reward

    if action == 'Sharp-Right-Turn':
        ss.heading_idx = (ss.heading_idx + 1) % 4
        step_dist = 0.05
    elif action == 'Slight-Right-Turn':
        step_dist = 0.10
    elif action == 'Slight-Left-Turn':
        ss.heading_idx = (ss.heading_idx - 1) % 4
        step_dist = 0.08
    else:
        step_dist = 0.15

    dx, dy = DIRECTIONS[ss.heading_idx]
    ss.x += dx * step_dist
    ss.y += dy * step_dist

    ss.trajectory.append({'x': ss.x, 'y': ss.y})

    ss.log.append(f"Step {ss.idx:04d} | State: {state:<10s} | Action: {action:<18s}")
    if len(ss.log) > 100:
        ss.log.pop(0)

    ss.idx += 1

# ---------------------------------------------------------------------
# 4. Streamlit Dashboard Layout
# ---------------------------------------------------------------------
st.title("📦 Warehouse AGV & Snake MDP Navigation")
st.markdown("Blueprint simulation showing the AGV cursor navigating warehouse storage aisles with entry/exit bays and a glowing snake trace path.")

col_left, col_right = st.columns([2.3, 1.0])

with col_left:
    btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 2])
    if btn1.button("▶ Play / Pause"):
        st.session_state.running = not st.session_state.running
    if btn2.button("Step ⏭"):
        step_simulation()
    if btn3.button("↺ Reset"):
        reset_sim()
    sim_speed = btn4.slider("Playback Speed", 0.01, 0.2, 0.03, 0.01)
    
    plot_placeholder = st.empty()

with col_right:
    st.subheader("📊 AGV Status")
    m1, m2 = st.columns(2)
    m1.metric("Progress", f"{st.session_state.idx} / {max_steps}")
    m2.metric("Reward", st.session_state.cum_reward)

    current_row = DATA.iloc[min(st.session_state.idx, len(DATA)-1)]
    cur_state = classify_state(current_row['SD_left'])
    st.info(f"**Current State:** `{cur_state}`\n\n"
            f"• Left Distance: `{current_row['SD_left']:.2f}m`\n"
            f"• Front Distance: `{current_row['SD_front']:.2f}m`")

    st.subheader("📜 Event Log")
    st.code("\n".join(st.session_state.log[-12:]) if st.session_state.log else "—", language=None)

# ---------------------------------------------------------------------
# 5. Warehouse Blueprint & Snake Trace Rendering
# ---------------------------------------------------------------------
def render_warehouse():
    ss = st.session_state
    fig, ax = plt.subplots(figsize=(7, 5.8))
    
    # Architectural Blueprint Dark Theme
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.set_aspect('equal')

    # Warehouse Outer Walls
    ax.plot([0, 12, 12, 0, 0], [0, 0, 10, 10, 0], color='#38bdf8', linewidth=3.5, label="Warehouse Walls")

    # Entry and Exit Gates
    ax.plot([0, 0], [2, 4], color='#22c55e', linewidth=6, label="Entry Gate")
    ax.plot([12, 12], [6, 8], color='#ef4444', linewidth=6, label="Exit Gate")

    # Internal Fulfillment Storage Racks (Obstacles)
    racks = [
        ([2.5, 5.0, 5.0, 2.5, 2.5], [2.0, 2.0, 4.5, 4.5, 2.0]),
        ([7.0, 9.5, 9.5, 7.0, 7.0], [2.0, 2.0, 4.5, 4.5, 2.0]),
        ([2.5, 5.0, 5.0, 2.5, 2.5], [5.5, 5.5, 8.0, 8.0, 5.5]),
        ([7.0, 9.5, 9.5, 7.0, 7.0], [5.5, 5.5, 8.0, 8.0, 5.5]),
    ]
    for rx, ry in racks:
        ax.fill(rx, ry, color='#1e293b', edgecolor='#64748b', linewidth=1.5)
        ax.text(np.mean(rx), np.mean(ry), "STORAGE RACK", color='#64748b', fontsize=7, ha='center', va='center', fontweight='bold', alpha=0.7)

    # Snake Xenzia Style Glowing Trail (limited to trail_length for clean fading effect)
    if len(ss.trajectory) > 1:
        recent_traj = ss.trajectory[-trail_length:]
        tx = [p['x'] for p in recent_traj]
        ty = [p['y'] for p in recent_traj]
        
        # Draw fading snake body segments
        for i in range(len(tx) - 1):
            alpha_val = (i + 1) / len(tx)
            ax.plot(tx[i:i+2], ty[i:i+2], color='#38bdf8', linewidth=3.5, alpha=alpha_val)

    # AGV Cursor Head (Snake Head)
    ax.scatter([ss.x], [ss.y], color='#facc15', s=200, zorder=5, marker='s', edgecolors='white', linewidths=1.5, label="AGV Cursor")

    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 11)
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
