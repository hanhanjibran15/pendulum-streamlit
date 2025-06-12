import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import io

# --- Streamlit Interface ---
st.title("Simulasi Pendulum Terbalik dengan LQR")
st.write("Masukkan parameter sistem dan kondisi awal untuk memulai simulasi.")

# --- Input Box ---
M = float(st.text_input("Massa kereta (kg)", "1.0"))
m = float(st.text_input("Massa pendulum (kg)", "0.1"))
L = float(st.text_input("Panjang pendulum (m)", "0.5"))
g = float(st.text_input("Percepatan gravitasi (m/s^2)", "9.81"))

x0 = float(st.text_input("Posisi awal kereta (m)", "0.0"))
x_dot0 = float(st.text_input("Kecepatan awal kereta (m/s)", "0.0"))
theta0 = float(st.text_input("Sudut awal pendulum (rad)", "0.1"))
theta_dot0 = float(st.text_input("Kecepatan sudut awal pendulum (rad/s)", "0.0"))

t_final = float(st.text_input("Waktu simulasi (detik)", "10.0"))

# Tombol untuk menjalankan simulasi
if st.button("Jalankan Simulasi"):

    def nonlinear_pendulum_dynamics(t, state, F_applied):
        x, x_dot, theta, theta_dot = state
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)
        theta_dot_sq = theta_dot**2

        mass_matrix = np.array([
            [M + m, m * L * c_theta],
            [m * L * c_theta, m * L**2]
        ])

        rhs_vector = np.array([
            F_applied + m * L * theta_dot_sq * s_theta,
            m * g * L * s_theta
        ])

        try:
            accelerations = np.linalg.solve(mass_matrix, rhs_vector)
            x_ddot = accelerations[0]
            theta_ddot = accelerations[1]
        except np.linalg.LinAlgError:
            x_ddot = 0.0
            theta_ddot = 0.0

        return [x_dot, x_ddot, theta_dot, theta_ddot]

    # --- Linearization ---
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -m * g / M, 0],
        [0, 0, 0, 1],
        [0, 0, (M + m) * g / (M * L), 0]
    ])

    B = np.array([
        [0],
        [1 / M],
        [0],
        [-1 / (M * L)]
    ])

    Q = np.diag([100.0, 1.0, 1000.0, 100.0])
    R = np.array([[1.0]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    def closed_loop_dynamics(t, state):
        F_control = -K @ state
        return nonlinear_pendulum_dynamics(t, state, F_control[0])

    initial_state = np.array([x0, x_dot0, theta0, theta_dot0])
    t_eval = np.linspace(0, t_final, 500)
    sol = solve_ivp(closed_loop_dynamics, (0, t_final), initial_state, t_eval=t_eval, method='RK45')

    t = sol.t
    x_pos = sol.y[0]
    theta = sol.y[2]
    control_forces = np.array([-K @ sol.y[:, i] for i in range(sol.y.shape[1])]).flatten()

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    axs[0].plot(t, x_pos)
    axs[0].set_ylabel("Posisi Kereta (m)")
    axs[0].grid(True)

    axs[1].plot(t, np.degrees(theta))
    axs[1].set_ylabel("Sudut Pendulum (deg)")
    axs[1].grid(True)

    axs[2].plot(t, control_forces)
    axs[2].set_ylabel("Gaya Kontrol (N)")
    axs[2].set_xlabel("Waktu (s)")
    axs[2].grid(True)

    fig.tight_layout()

    st.pyplot(fig)

    # Export video hint (streamlit doesn't support video rendering directly from matplotlib animation)
    st.info("Animasi hanya tersedia di versi lokal Python, bukan di Streamlit web. Gunakan matplotlib animation seperti di kode utama Anda untuk menyimpannya sebagai video.")
