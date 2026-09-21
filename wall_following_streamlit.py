"""
Streamlit Dashboard: MDP Wall-Following Robot Navigation (Closed Loop Room Layout)
================================================================================
Run with:
    streamlit run app.py
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDP Wall-Following Robot Dashboard", layout="wide")

# ---------------------------------------------------------------------
# 1. Load Data & MDP Policies
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

# MDP Reward Matrix
R = {
    'Too-Close': {'Move-Forward': -10, 'Slight-Right-Turn': 5, 'Sharp-Right-Turn': 10, 'Slight-Left-Turn': -10},
    'Ideal':     {'Move-Forward': 10, 'Slight-Right-Turn': 2, 'Sharp-Right-Turn': -5, 'Slight-Left-Turn': 2},
    'Too-Far':   {'Move-Forward': -2, 'Slight-Right-Turn': -10, 'Sharp-Right-Turn': -10, 'Slight-Left-Turn': 10},
}

# ---------------------------------------------------------------------
# 2. Sidebar Controls
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Simulation Parameters")
too_close_thresh = st.sidebar.slider("Too-Close Threshold (m)", 0.3, 0.6, 0.53, 0.01)
too_far_thresh = st.sidebar.slider("Too-Far Threshold (m)", 0.6, 1.2, 0.71, 0.01)
max_steps = st.sidebar.slider("Dataset Steps to Simulate", 500, len(DATA), 2000, 100)

def classify_state(sd_left):
    if sd_left < too_close_thresh:
        return 'Too-Close'
    elif sd_left > too_far_thresh:
        return 'Too-Far'
    else:
        return 'Ideal'

# ---------------------------------------------------------------------
# 3. Session State Initialization (Rectangular Room Layout)
# ---------------------------------------------------------------------
def reset_sim():
    st.session_state.idx = 0
    st.session_state.x = 2.0
    st.session_state.y = 8.0
    st.session_state.heading_idx = 0  # 0: East, 1: South, 2: West, 3: North (Clockwise)
    st.session_state.cum_reward = 0
    st.session_state.trajectory = [{'x': 2.0, 'y': 8.0}]
    st.session_state.wall_points = []
    st.session_state.log = []
    st.session_state.running = False
    st.session_state.finished = False

if 'idx' not in st.session_state:
    reset_sim()

# Direction vectors for clockwise rectangular room navigation
DIRECTIONS = [
    (1.0, 0.0),   # East
    (0.0, -1.0),  # South
    (-1.0, 0.0),  # West
    (0.0, 1.0)    # North
]

def step_simulation():
    ss = st.session_state
    if ss.idx >= max_steps:
        ss.finished = True
        ss.running = False
        return

    row = DATA.iloc[ss.idx]
    sd_left = row['SD_left']
    dataset_action = row['Class']

    # Determine state and action from MDP policy
    state = classify_state(sd_left)
    action = POLICY.get(state, dataset_action)
    if action not in R[state]:
        action = 'Move-Forward'

    reward = R[state][action]
    ss.cum_reward += reward

    # Update heading and position based on action & rectangular room constraints
    if action == 'Sharp-Right-Turn':
        ss.heading_idx = (ss.heading_idx + 1) % 4
        step_dist = 0.05
    elif action == 'Slight-Right-Turn':
        step_dist = 0.10
    elif action == 'Slight-Left-Turn':
        ss.heading_idx = (ss.heading_idx - 1) % 4
        step_dist = 0.08
    else:  # Move-Forward
        step_dist = 0.15

    dx, dy = DIRECTIONS[ss.heading_idx]
    ss.x += dx * step_dist
    ss.y += dy * step_dist

    ss.trajectory.append({'x': ss.x, 'y': ss.y})

    # Calculate adjacent wall point (to the left of heading vector)
    left_dir_idx = (ss.heading_idx - 1) % 4
    wx_dir, wy_dir = DIRECTIONS[left_dir_idx]
    wall_x = ss.x + wx_dir * sd_left
    wall_y = ss.y + wy_dir * sd_left
    ss.wall_points.append({'x': wall_x, 'y': wall_y})

    ss.log.append(f"Step {ss.idx:04d} | State: {state:<10s} | Action: {action:<18s}")
    if len(ss.log) > 100:
        ss.log.pop(0)

    ss.idx += 1

# ---------------------------------------------------------------------
# 4. Main Dashboard UI
# ---------------------------------------------------------------------
st.title("🤖 MDP Wall-Following Robot Navigation")
st.markdown("Simulating the robot's **rectangular room circuit** and adjacent wall traces using the MDP policy[cite: 3, 4].")

col_left, col_right = st.columns([2.1, 1.0])

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
    st.subheader("📊 Status")
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
# 5. Plotting Function (Closed Circuit Room & Wall Trace)
# ---------------------------------------------------------------------
def render_visualization():
    ss = st.session_state
    fig, ax = plt.subplots(figsize=(7, 5.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f9fa')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6, color='#dddddd')

    # Draw room boundary outline (fixed reference rectangular room)
    ax.plot([0, 10, 10, 0, 0], [0, 0, 10, 10, 0], color='#adb5bd', linestyle='--', linewidth=1.5, label="Room Boundaries")

    # Plot Wall Trace mapped adjacent to robot
    if ss.wall_points:
        wx = [p['x'] for p in ss.wall_points]
        wy = [p['y'] for p in ss.wall_points]
        ax.scatter(wx, wy, color="#e63946", s=10, alpha=0.7, label="Mapped Adjacent Wall")

    # Plot Robot Trajectory Loop
    if len(ss.trajectory) > 1:
        tx = [p['x'] for p in ss.trajectory]
        ty = [p['y'] for p in ss.trajectory]
        ax.plot(tx, ty, color="#1d3557", linewidth=2.5, label="Robot Path", zorder=3)

    # Plot Robot Position
    dx, dy = DIRECTIONS[ss.heading_idx]
    ax.scatter([ss.x], [ss.y], color="#457b9d", s=150, zorder=4, edgecolors='black', linewidths=1.2, label="SCITOS Robot")
    ax.arrow(ss.x, ss.y, dx*0.4, dy*0.4, head_width=0.2, head_length=0.25, fc='#e63946', ec='black', zorder=5)

    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.set_title("Robot Rectangular Circuit & Wall-Following Path", fontsize=11, fontweight='bold', pad=10)
    
    plot_placeholder.pyplot(fig, use_container_width=True)
    plt.close(fig)

render_visualization()

if st.session_state.running and not st.session_state.finished:
    for _ in range(15):
        if not st.session_state.running or st.session_state.finished:
            break
        step_simulation()
        render_visualization()
        time.sleep(sim_speed)
    st.rerun()
