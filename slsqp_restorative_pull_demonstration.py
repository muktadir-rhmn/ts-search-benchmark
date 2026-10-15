"""
SLSQP Restorative Pull Demonstration (Interactive)
Before Convergence, Intermediate Points Can Violate Constraints. 
But after convergence, the final point P2 will satisfy the constraint c(x, y) = 0 (i.e., lie on the unit circle).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 1. Setup Figure and Axes
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.25) # Make room at the bottom for sliders

# Static limits for a stable view while sliding
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle=':', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title("SLSQP Restorative Pull (Interactive)")
ax.set_xlabel("x")
ax.set_ylabel("y")

# 2. Plot Static True Constraint (Unit Circle)
theta = np.linspace(0, 2 * np.pi, 300)
x_circle = np.cos(theta)
y_circle = np.sin(theta)
ax.plot(x_circle, y_circle, 'b-', linewidth=2, label="True Constraint ($x^2 + y^2 = 1$)")

# 3. Initial Values
xk_init = -0.5
yk_init = 1.0

# SLSQP's QP subproblem (solved by its LSQ/LSEI routine):
#   min_d  0.5 d^T B d + g^T d   s.t.  A d + c = 0
# For equality constraints this is the KKT system
#   [B  A^T] [d     ]   [-g]
#   [A   0 ] [lambda] = [-c]
def solve_qp_subproblem(B, g, A, c):
    n, m = B.shape[0], A.shape[0]
    K = np.block([[B, A.T], [A, np.zeros((m, m))]])
    rhs = np.concatenate([-g, -c])
    sol = np.linalg.lstsq(K, rhs, rcond=None)[0] # lstsq handles a singular KKT matrix (grad c = 0)
    return sol[:n]

# Function to calculate the linearized constraint and the final point
def calc_values(xk, yk):
    yk_safe = yk if abs(yk) > 1e-5 else 1e-5 # Avoid division by zero
    x_line = np.linspace(xk - 1.5, xk + 1.5, 100)
    y_line = (xk**2 + yk_safe**2 + 1 - 2 * xk * x_line) / (2 * yk_safe)

    # Pure restoration step: no objective (g = 0) and SLSQP's initial Hessian approximation B = I
    B = np.eye(2)
    g = np.zeros(2)
    A = np.array([[2 * xk, 2 * yk]])       # Jacobian of c(x, y) = x^2 + y^2 - 1
    c = np.array([xk**2 + yk**2 - 1])
    d = solve_qp_subproblem(B, g, A, c)
    x_final, y_final = xk + d[0], yk + d[1]
    return x_line, y_line, x_final, y_final

x_line_init, y_line_init, x_final_init, y_final_init = calc_values(xk_init, yk_init)

# 4. Initialize Dynamic Plot Elements
line, = ax.plot(x_line_init, y_line_init, 'r--', linewidth=2, label="Linearized Constraint")
p1_scatter = ax.scatter(xk_init, yk_init, color="red", zorder=5, s=100, label="Initial Point P1")
p2_scatter = ax.scatter(x_final_init, y_final_init, color="green", zorder=5, s=100, label="Final Point P2")

# Annotations (arrows) are tricky to update, so we store it in a list to remove and redraw it
arrow_ref = [ax.annotate('', xy=(x_final_init, y_final_init), xytext=(xk_init, yk_init),
                         arrowprops=dict(facecolor='black', shrink=0, width=2, headwidth=8))]

ax.legend(loc="upper right")

# 5. Create Slider Axes and Sliders
ax_x = fig.add_axes([0.2, 0.1, 0.65, 0.03])
ax_y = fig.add_axes([0.2, 0.05, 0.65, 0.03])

slider_x = Slider(ax_x, 'x', -2.0, 2.0, valinit=xk_init)
slider_y = Slider(ax_y, 'y', -2.0, 2.0, valinit=yk_init)

# 6. Update Function for the Sliders
def update(val):
    xk = slider_x.val
    yk = slider_y.val
    
    # Recalculate based on new slider values
    x_line, y_line, x_final, y_final = calc_values(xk, yk)
    
    # Update line
    line.set_xdata(x_line)
    line.set_ydata(y_line)
    
    # Update points
    p1_scatter.set_offsets(np.c_[xk, yk])
    p2_scatter.set_offsets(np.c_[x_final, y_final])
    
    # Update arrow (remove old, draw new)
    arrow_ref[0].remove()
    # Only draw the arrow if the points aren't exactly the same to avoid matplotlib errors
    if np.hypot(x_final - xk, y_final - yk) > 1e-4:
        arrow_ref[0] = ax.annotate('', xy=(x_final, y_final), xytext=(xk, yk),
                                   arrowprops=dict(facecolor='black', shrink=0, width=2, headwidth=8))
    else:
        arrow_ref[0] = ax.annotate('', xy=(xk, yk), xytext=(xk, yk)) # Blank dummy annotation
        
    fig.canvas.draw_idle()

# Register the update function with the sliders
slider_x.on_changed(update)
slider_y.on_changed(update)

# 7. Next / Play Buttons: move P1 to the current P2 and re-solve the QP subproblem
ax_next = fig.add_axes([0.55, 0.15, 0.13, 0.05])
ax_play = fig.add_axes([0.72, 0.15, 0.13, 0.05])
btn_next = Button(ax_next, 'Next')
btn_play = Button(ax_play, 'Play')

def step():
    _, _, x_final, y_final = calc_values(slider_x.val, slider_y.val)
    converged = np.hypot(x_final - slider_x.val, y_final - slider_y.val) < 1e-8
    # Set x silently so update() runs once, with both new coordinates
    slider_x.eventson = False
    slider_x.set_val(x_final)
    slider_x.eventson = True
    slider_y.set_val(y_final)
    return converged

def on_next(event):
    step()

# Timer drives continuous stepping while playing
timer = fig.canvas.new_timer(interval=700)

def on_tick():
    if step():
        stop_play()

def stop_play():
    timer.stop()
    btn_play.label.set_text('Play')
    fig.canvas.draw_idle()

def on_play(event):
    if btn_play.label.get_text() == 'Play':
        btn_play.label.set_text('Pause')
        timer.start()
        fig.canvas.draw_idle()
    else:
        stop_play()

timer.add_callback(on_tick)
btn_next.on_clicked(on_next)
btn_play.on_clicked(on_play)

plt.show()