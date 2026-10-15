"""Interactive Matplotlib GUI for comparing FSM and GSM.

Run with ``python test.py``. Tick "Animate" to watch each selected
algorithm build its path step by step (FSM first, then GSM) before the final
paths are drawn. "Play" replays the steps on a timer; "Step" waits for the
Next/Back buttons. Switching between them mid-animation pauses or resumes it.
"""

from __future__ import annotations

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from scipy.interpolate import PchipInterpolator

from surfaces import leps
from surfaces import muller_brown
from algorithms.fsm import FSMStep, freezing_string
from algorithms.gsm import GSMStep, growing_string
from algorithms.string_utils import FSM_NODE_OPTIMIZERS, GSM_STRING_OPTIMIZERS, TS_REFINEMENTS


SURFACES = {
    "Muller-Brown": {
        "energy": muller_brown.energy,
        "gradient": muller_brown.gradient,
        "reactant": muller_brown.DEFAULT_REACTANT,
        "product": muller_brown.DEFAULT_PRODUCT,
        "bounds": muller_brown.PLOT_BOUNDS,
        "levels": np.linspace(-145.0, 120.0, 45),
        "labels": ("x", "y"),
    },
    "LEPS (collinear)": {
        "energy": leps.energy,
        "gradient": leps.gradient,
        "reactant": leps.DEFAULT_REACTANT,
        "product": leps.DEFAULT_PRODUCT,
        "bounds": leps.PLOT_BOUNDS,
        "levels": np.linspace(-4.8, 2.0, 45),
        "labels": (r"$r_{AB}$ / Angstrom", r"$r_{BC}$ / Angstrom"),
    },
}

APP_VERSION = "1.7.0"

FSM_COLOR = "#ff7f0e"
GSM_COLOR = "#e7298a"
REACTANT_FRONT_COLOR = "#00e5ff"
PRODUCT_FRONT_COLOR = "#7CFC00"

# GSM can take hundreds of iterations (two frames each: evolve, reparametrize).
# Up to this many iterations every one is shown; beyond it, the first
# iteration after each growth plus a log-spaced subset of the rest.
MAX_GSM_ITERATION_FRAMES = 60


def format_counts(source: object) -> str:
    """Energy/gradient usage of a result or step snapshot, for the status text."""

    return (
        f"energy: {source.energy_calls} calls ({source.energy_evaluations} pts), "
        f"gradient: {source.gradient_calls} calls ({source.gradient_evaluations} pts)"
    )


def display_curve(images: np.ndarray, samples_per_segment: int = 16) -> np.ndarray:
    """Return a smooth display curve that passes through every path image."""

    segment_lengths = np.linalg.norm(np.diff(images, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if arc[-1] == 0.0 or np.any(np.diff(arc) <= 0.0):
        return images.copy()
    # Every original arc coordinate is included exactly, so the displayed
    # curve contains every image, including the indexed TS image.
    sample_arc = np.concatenate(
        [
            np.linspace(arc[i], arc[i + 1], samples_per_segment, endpoint=False)
            for i in range(len(images) - 1)
        ]
        + [arc[-1:]]
    )
    return np.column_stack(
        [PchipInterpolator(arc, images[:, axis])(sample_arc) for axis in range(images.shape[1])]
    )


class Dropdown:
    """Minimal combo box (Matplotlib has none): a button showing the current
    choice that opens the list of options below it; click one to select it.

    The list is an axes drawn above the widgets it covers. Matplotlib widgets
    hit-test their own axes regardless of what is drawn on top, so while the
    list is open every widget in ``blocked`` is deactivated; otherwise a click
    on an option would also press the button or slider underneath. Clicking
    anywhere else closes the list.
    """

    def __init__(self, figure, rect, options, redraw, initial: int = 0) -> None:
        self.figure = figure
        self.options = tuple(options)
        self.index = initial
        self.redraw = redraw
        self.axis = figure.add_axes(rect)
        self.button = Button(self.axis, self.options[initial] + "  ▾", color="white", hovercolor="#e8eef7")
        self.button.label.set_fontsize(8.5)
        self.button.on_clicked(self.toggle)

        x, y, width, height = rect
        rows = len(self.options)
        self.list_axis = figure.add_axes((x, y - rows * height, width, rows * height))
        self.list_axis.set_zorder(100)  # above the widgets it covers
        self.list_axis.set(xlim=(0, 1), ylim=(rows, 0), xticks=[], yticks=[])
        self.highlight = self.list_axis.axhspan(0, 1, color="#cfe0f5")
        for row, option in enumerate(self.options):
            self.list_axis.text(0.04, row + 0.5, option, va="center", fontsize=8.5)
        self.list_axis.set_visible(False)
        self.blocked: list = []  # set by the owner once all widgets exist
        self.unblock_on_release = False
        figure.canvas.mpl_connect("button_press_event", self.on_press)
        figure.canvas.mpl_connect("button_release_event", self.on_release)

    @property
    def value(self) -> str:
        return self.options[self.index]

    def toggle(self, _event: object = None) -> None:
        opening = not self.list_axis.get_visible()
        if opening:
            self.highlight.set_y(self.index)  # axhspan is a Rectangle in 3.9
        self.list_axis.set_visible(opening)
        self.set_blocked(opening)
        self.redraw()

    def set_blocked(self, block: bool) -> None:
        for widget in self.blocked:
            # Not set_active(): Radio/CheckButtons override it to select an option.
            widget.active = not block

    def on_press(self, event) -> None:
        if not self.list_axis.get_visible() or event.inaxes is self.axis:
            return  # the button's own release toggles the list
        if event.inaxes is self.list_axis and event.ydata is not None:
            self.index = int(np.clip(event.ydata, 0, len(self.options) - 1))
            self.button.label.set_text(self.value + "  ▾")
        self.list_axis.set_visible(False)
        # Keep the covered widgets inactive until this click is released:
        # widgets connected after this one still receive the same press.
        self.unblock_on_release = True
        self.redraw()

    def on_release(self, _event) -> None:
        if self.unblock_on_release:
            self.unblock_on_release = False
            self.set_blocked(False)


class StringMethodGUI:
    def __init__(self) -> None:
        self.surface_name = "Muller-Brown"
        self.results = []
        # Animation state: recorded FSM/GSM steps, the frame queue, the running
        # timer, and the temporary artists drawn for the current frame.
        self.fsm_steps: list[FSMStep] = []
        self.gsm_steps: list[GSMStep] = []
        # Each frame is ("FSM", FSMStep, "seed" | "freeze") or ("GSM", GSMStep, "").
        self.frames: list[tuple[str, FSMStep | GSMStep, str]] = []
        self.frame_index = 0  # index of the next frame to draw
        self.animating = False  # frames are on screen (playing or paused)
        self.timer = None
        self.frame_artists: list = []
        self.figure, self.axis = plt.subplots(figsize=(10.5, 8.2))
        self.figure.subplots_adjust(left=0.28, right=0.97, bottom=0.17, top=0.95)

        surface_axis = self.figure.add_axes((0.025, 0.80, 0.21, 0.11))
        self.surface_buttons = RadioButtons(surface_axis, tuple(SURFACES), active=0)
        self.surface_buttons.on_clicked(self.change_surface)
        surface_axis.set_title("Energy surface", fontsize=10)

        algorithm_axis = self.figure.add_axes((0.025, 0.68, 0.21, 0.08))
        self.algorithm_buttons = CheckButtons(algorithm_axis, ("FSM", "GSM"), (True, True))
        algorithm_axis.set_title("Algorithms", fontsize=10)

        # Optimizer combo boxes; an open list drops down over the widgets below.
        combos = (
            ("ts", "TS refinement", TS_REFINEMENTS, 0.475),
            ("gsm", "GSM string optimizer", GSM_STRING_OPTIMIZERS, 0.545),
            ("fsm", "FSM node optimizer", FSM_NODE_OPTIMIZERS, 0.615),
        )
        self.optimizer_menus: dict[str, Dropdown] = {}
        for key, title, options, bottom in combos:
            self.figure.text(0.03, bottom + 0.043, title, fontsize=9, va="center")
            self.optimizer_menus[key] = Dropdown(self.figure, (0.025, bottom, 0.21, 0.03), options, self.refresh)

        slider_axis = self.figure.add_axes((0.04, 0.395, 0.18, 0.03))
        self.figure.text(0.04, 0.44, "Images", fontsize=9, va="center")
        self.image_slider = Slider(slider_axis, "", 9, 41, valinit=25, valstep=2)

        run_axis = self.figure.add_axes((0.055, 0.32, 0.15, 0.05))
        self.run_button = Button(run_axis, "Run")
        self.run_button.on_clicked(self.run_algorithms)

        animate_axis = self.figure.add_axes((0.025, 0.25, 0.105, 0.055))
        self.animate_buttons = CheckButtons(animate_axis, ("Animate",), (True,))

        # Play on a timer, or Step through the frames with Next/Back.
        mode_axis = self.figure.add_axes((0.13, 0.25, 0.105, 0.055))
        self.mode_buttons = RadioButtons(mode_axis, ("Play", "Step"), active=0)
        self.mode_buttons.on_clicked(self.change_mode)

        delay_axis = self.figure.add_axes((0.04, 0.2, 0.18, 0.022))
        self.figure.text(0.04, 0.235, "Frame delay (ms)", fontsize=9, va="center")
        self.delay_slider = Slider(delay_axis, "", 50, 1500, valinit=400, valstep=50)
        self.delay_slider.on_changed(self.change_delay)

        back_axis = self.figure.add_axes((0.04, 0.14, 0.085, 0.045))
        self.back_button = Button(back_axis, "◂ Back")
        self.back_button.on_clicked(self.previous_frame)
        next_axis = self.figure.add_axes((0.135, 0.14, 0.085, 0.045))
        self.next_button = Button(next_axis, "Next ▸")
        self.next_button.on_clicked(self.next_frame)

        widgets = [self.surface_buttons, self.algorithm_buttons, self.image_slider, self.run_button,
                   self.animate_buttons, self.mode_buttons, self.delay_slider,
                   self.back_button, self.next_button]
        widgets += [menu.button for menu in self.optimizer_menus.values()]
        for menu in self.optimizer_menus.values():
            menu.blocked = [widget for widget in widgets if widget is not menu.button]

        self.status = self.figure.text(
            0.025, 0.015, f"v{APP_VERSION} — Choose a surface and click Run.",
            fontsize=9, va="bottom", wrap=True,
        )
        self.draw_surface()

    def refresh(self, flush: bool = True) -> None:
        """Repaint the window now.

        ``draw_idle`` only *requests* a repaint, and on some macOS setups those
        requests are not honoured after a button callback, leaving the window
        frozen on "Computing paths..." while the animation runs unseen. A full
        draw plus ``flush_events`` makes every frame reach the screen.

        Animation frames pass ``flush=False``: they run from a timer, so the
        event loop paints them as soon as the callback returns. Flushing there
        would process user input (e.g. a slider drag) inside the timer callback,
        and the stream of drag events can lock up the window.
        """

        self.figure.canvas.draw()
        if flush:
            self.figure.canvas.flush_events()

    @property
    def selected_algorithms(self) -> list[str]:
        return [
            label.get_text()
            for label, active in zip(self.algorithm_buttons.labels, self.algorithm_buttons.get_status())
            if active
        ]

    @property
    def animate(self) -> bool:
        return bool(self.animate_buttons.get_status()[0])

    @property
    def stepping(self) -> bool:
        return self.mode_buttons.value_selected == "Step"

    def change_surface(self, name: str) -> None:
        self.stop_animation()
        self.surface_name = name
        self.results.clear()
        self.draw_surface()

    def draw_surface(self) -> None:
        self.axis.clear()
        spec = SURFACES[self.surface_name]
        (xmin, xmax), (ymin, ymax) = spec["bounds"]
        x = np.linspace(xmin, xmax, 260)
        y = np.linspace(ymin, ymax, 260)
        xx, yy = np.meshgrid(x, y)
        points = np.stack((xx, yy), axis=-1)
        zz = np.asarray(spec["energy"](points))
        levels = spec["levels"]
        clipped = np.clip(zz, levels[0], levels[-1])
        self.axis.contourf(xx, yy, clipped, levels=levels, cmap="viridis")
        self.axis.contour(xx, yy, clipped, levels=levels[::3], colors="white", alpha=0.25, linewidths=0.5)
        reactant, product = spec["reactant"], spec["product"]
        self.axis.scatter(*reactant, c="cyan", edgecolor="black", s=75, label="Reactant", zorder=5)
        self.axis.scatter(*product, c="lime", edgecolor="black", s=75, label="Product", zorder=5)
        self.axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel=spec["labels"][0], ylabel=spec["labels"][1])
        # Equal data scaling on both axes, so right angles (e.g. the FSM constraint
        # segment vs. the growth direction) look like right angles.
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.set_title(f"{self.surface_name} potential — v{APP_VERSION}")
        self.axis.legend(loc="upper right", fontsize=8)
        self.refresh()

    def run_algorithms(self, _event: object) -> None:
        if self.animating:
            # The button reads "Stop" while animating: skip to the final paths.
            self.stop_animation()
            self.draw_results()
            return
        selected = self.selected_algorithms
        if not selected:
            self.status.set_text("Select FSM, GSM, or both.")
            self.refresh()
            return
        self.stop_animation()
        self.run_button.label.set_text("Running...")
        self.status.set_text("Computing paths...")
        self.refresh()
        try:
            self.compute_paths(selected)
            if self.animate and (self.fsm_steps or self.gsm_steps):
                self.start_animation()
            else:
                self.draw_results()
        except Exception as error:
            # Matplotlib swallows exceptions raised in widget callbacks (they
            # only reach the terminal), which would leave the window stuck on
            # "Computing paths...". Show the error and re-enable the button.
            traceback.print_exc()
            self.stop_animation()
            self.status.set_text(f"Error: {type(error).__name__}: {error} (full traceback in the terminal)")
            self.run_button.label.set_text("Run")
            self.refresh()

    def compute_paths(self, selected: list[str]) -> None:
        spec = SURFACES[self.surface_name]
        count = int(self.image_slider.val)
        self.results.clear()
        self.fsm_steps = []
        self.gsm_steps = []
        animate = self.animate
        algorithms = {"FSM": freezing_string, "GSM": growing_string}
        recorders = {"FSM": self.fsm_steps.append, "GSM": self.gsm_steps.append}
        ts_refinement = self.optimizer_menus["ts"].value
        options = {
            "FSM": {"node_optimizer": self.optimizer_menus["fsm"].value, "ts_refinement": ts_refinement},
            "GSM": {"string_optimizer": self.optimizer_menus["gsm"].value, "ts_refinement": ts_refinement},
        }

        for method in selected:
            # Progress goes to the terminal so a stall shows where it happened.
            chosen = ", ".join(options[method].values())
            print(f"[v{APP_VERSION}] {method} ({chosen}) on {self.surface_name}, {count} images ...",
                  end=" ", flush=True)
            started = time.perf_counter()
            self.results.append(
                algorithms[method](
                    spec["energy"], spec["gradient"], spec["reactant"], spec["product"],
                    num_images=count, bounds=spec["bounds"],
                    # Record every step only when the animation is wanted.
                    callback=recorders[method] if animate else None,
                    **options[method],
                )
            )
            result = self.results[-1]
            print(
                f"done in {time.perf_counter() - started:.2f} s; {format_counts(result)}",
                flush=True,
            )

    # --- Animation ----------------------------------------------------------
    def start_animation(self) -> None:
        """Replay the recorded FSM then GSM steps, then draw the final results."""

        self.draw_surface()
        # FSM: two frames per growth step, first where the seed lands and the
        # constraint it must obey, then the optimizer moving it and freezing.
        self.frames = [("FSM", step, phase) for step in self.fsm_steps for phase in ("seed", "freeze")]
        self.frames += [("GSM", step, "") for step in self.gsm_animation_steps()]
        self.frame_index = 0
        self.animating = True
        self.run_button.label.set_text("Stop")
        if self.stepping:
            self.advance_frame()  # show the first frame, then wait for Next
            self.figure.canvas.flush_events()
        else:
            self.start_timer()

    def start_timer(self) -> None:
        self.timer = self.figure.canvas.new_timer(interval=int(self.delay_slider.val))
        self.timer.add_callback(self.advance_frame)
        self.timer.start()

    def stop_timer(self) -> None:
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    def change_mode(self, label: str) -> None:
        # Mid-animation, Step pauses on the current frame and Play resumes.
        if not self.animating:
            return
        if label == "Step":
            self.stop_timer()
        elif self.timer is None:
            self.start_timer()

    def next_frame(self, _event: object) -> None:
        if not self.animating:
            return
        self.stop_timer()  # stepping by hand while playing pauses playback
        self.advance_frame()
        self.figure.canvas.flush_events()

    def previous_frame(self, _event: object) -> None:
        # The current frame is frame_index - 1; redraw the one before it.
        if not self.animating or self.frame_index < 2:
            return
        self.stop_timer()
        self.frame_index -= 2
        self.advance_frame()
        self.figure.canvas.flush_events()

    def change_delay(self, value: float) -> None:
        # Some backends (macOS) ignore a new interval on a running timer, so
        # restart it. This runs from the slider callback, never mid-frame.
        if self.timer is not None:
            self.timer.stop()
            self.timer.interval = int(value)
            self.timer.start()

    def stop_animation(self) -> None:
        self.stop_timer()
        self.animating = False
        self.clear_frame_artists()

    def clear_frame_artists(self) -> None:
        for artist in self.frame_artists:
            artist.remove()
        self.frame_artists = []

    def advance_frame(self) -> None:
        if self.frame_index >= len(self.frames):
            self.stop_animation()
            self.draw_results()
            return
        method, step, phase = self.frames[self.frame_index]
        if method == "FSM":
            self.draw_fsm_frame(step, phase)
        else:
            self.draw_gsm_frame(step)
        self.frame_index += 1
        if self.timer is None:
            # Stepping by hand: say where we are and what the buttons do.
            self.status.set_text(self.status.get_text() + f"\nFrame {self.frame_index}/{len(self.frames)}"
                                 " — Next ▸ to advance, ◂ Back to go back, Stop to skip to the result.")
            self.refresh(flush=False)

    def gsm_animation_steps(self) -> list[GSMStep]:
        """Every growth, plus the evolve/reparametrize pair of selected iterations."""

        iterations = sorted({step.iteration for step in self.gsm_steps if step.stage != "grow"})
        keep = set(iterations)
        if len(iterations) > MAX_GSM_ITERATION_FRAMES:
            # Always the step right after a growth (the new node first moving)
            # and the last step; fill up with log-spaced others.
            keep = {step.iteration for step in self.gsm_steps if step.stage == "grow"} | {iterations[-1]}
            spare = max(MAX_GSM_ITERATION_FRAMES - len(keep), 1)
            picks = np.unique(np.geomspace(1, len(iterations), spare).astype(int) - 1)
            keep |= {iterations[i] for i in picks}
        return [step for step in self.gsm_steps if step.stage == "grow" or step.iteration in keep]

    def draw_fsm_frame(self, step: FSMStep, phase: str) -> None:
        self.clear_frame_artists()
        plot = self.axis.plot
        reactant_side, product_side = step.reactant_side, step.product_side
        if phase == "seed":
            # The new image is not frozen yet: drop it from its side.
            if step.side == "reactant":
                reactant_side = reactant_side[:-1]
            else:
                product_side = product_side[:-1]
        growing, opposite = (
            (reactant_side[-1], product_side[-1]) if step.side == "reactant"
            else (product_side[-1], reactant_side[-1])
        )

        artists = [
            *plot(reactant_side[:, 0], reactant_side[:, 1], "o-", color=REACTANT_FRONT_COLOR,
                  markersize=4, linewidth=1.5, zorder=6, label="Reactant front (frozen)"),
            *plot(product_side[:, 0], product_side[:, 1], "o-", color=PRODUCT_FRONT_COLOR,
                  markersize=4, linewidth=1.5, zorder=6, label="Product front (frozen)"),
        ]
        if phase == "seed":
            # Gap between the two frontiers; the seed sits on this line.
            artists += plot(*np.column_stack((growing, opposite)), "--", color="white",
                            linewidth=1.0, alpha=0.8, zorder=5)
            if step.seed_image.size == 2:
                # In 2-D the allowed region is a segment through the seed,
                # perpendicular to the growth direction, of half-length = trust radius.
                normal = np.array([-step.growth_direction[1], step.growth_direction[0]])
                ends = np.vstack((step.seed_image - step.trust_radius * normal,
                                  step.seed_image + step.trust_radius * normal))
                artists += plot(ends[:, 0], ends[:, 1], ":", color=FSM_COLOR, linewidth=1.8,
                                zorder=7, label="Allowed moves (⊥ growth, trust radius)")
                # Debug: the constraint segment should be orthogonal to the growth direction.
                gap_direction = (opposite - growing) / np.linalg.norm(opposite - growing)
                segment_direction = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
                print(
                    f"[FSM debug] {step.side} side, frame {self.frame_index + 1}:\n"
                    f"  growing frontier  = {growing}\n"
                    f"  opposite frontier = {opposite}\n"
                    f"  growth direction  = {step.growth_direction}\n"
                    f"  gap direction     = {gap_direction} (drawn dashed line)\n"
                    f"  seed              = {step.seed_image}, trust radius = {step.trust_radius:.4g}\n"
                    f"  segment ends      = {ends[0]} -> {ends[1]}\n"
                    f"  segment direction = {segment_direction}\n"
                    f"  dot(segment, growth) = {segment_direction @ step.growth_direction:.3e}, "
                    f"dot(segment, gap) = {segment_direction @ gap_direction:.3e}\n"
                    f"  axis aspect = {self.axis.get_aspect()}, "
                    f"xlim = {self.axis.get_xlim()}, ylim = {self.axis.get_ylim()}",
                    flush=True,
                )
            artists.append(self.axis.scatter(*step.seed_image, marker="x", s=70, color=FSM_COLOR,
                                             linewidths=2.0, zorder=8, label="Seed"))
            message = f"seed placed {np.round(step.seed_image, 3)}"
        else:
            path = step.optimizer_trajectory
            artists += plot(path[:, 0], path[:, 1], ".-", color=FSM_COLOR, linewidth=1.2,
                            markersize=5, zorder=7, label="Optimizer iterates")
            artists.append(self.axis.scatter(*step.new_image, s=90, color=FSM_COLOR,
                                             edgecolor="black", zorder=9, label="New frozen image"))
            moved = float(np.linalg.norm(step.new_image - step.seed_image))
            status = "converged" if step.image_converged else "NOT converged"
            message = f"optimized in {len(path) - 1} moves, shifted {moved:.3f} from seed ({status})"
            low, high = np.array(SURFACES[self.surface_name]["bounds"]).T
            if np.any((path <= low + 1.0e-6) | (path >= high - 1.0e-6)):
                # The optimizer may take a long trial step that gets clipped to the
                # bounds before it settles back near the seed.
                message += "; an iterate was clipped at the box edge"

        self.frame_artists = artists
        self.frame_artists.append(self.axis.legend(loc="upper right", fontsize=8))
        # The seed frame comes before this step's optimization, so its running
        # totals are those left by the previous step.
        if phase != "seed":
            counts = format_counts(step)
        elif step.step > 0:
            counts = format_counts(self.fsm_steps[step.step - 1])
        else:
            counts = "no energy or gradient calls yet"
        self.status.set_text(
            f"FSM step {step.step + 1}/{len(self.fsm_steps)} — growing {step.side} side: {message}\n"
            f"So far: {counts}"
        )
        self.refresh(flush=False)

    def draw_gsm_frame(self, step: GSMStep) -> None:
        """Draw one GSM stage in the style of Peters et al. Fig. 2: the string
        before the stage as a ghost, the string after it on top."""

        self.clear_frame_artists()
        plot = self.axis.plot
        artists = []
        images, previous = step.images, step.previous_images
        joined = step.reactant_count is None
        if joined:
            phase = f"joined string, step {step.iteration}"
        else:
            phase = f"growing ({step.addition + 1}/{step.total_additions} nodes added), step {step.iteration}"

        def draw_string(points, count, style, **kwargs):
            # The two fragments in their front colors while growing, one
            # colour once joined; the vacant gap as a dashed white line.
            if count is None:
                return plot(points[:, 0], points[:, 1], style, color=GSM_COLOR, **kwargs)
            drawn = plot(points[:count, 0], points[:count, 1], style, color=REACTANT_FRONT_COLOR, **kwargs)
            drawn += plot(points[count:, 0], points[count:, 1], style, color=PRODUCT_FRONT_COLOR, **kwargs)
            gap = points[count - 1: count + 1]
            drawn += plot(gap[:, 0], gap[:, 1], "--", color="white", linewidth=1.0, alpha=0.8, zorder=5)
            return drawn

        def node_shifts(before, after, color):
            # A short segment from each node's old to its new position.
            segments = []
            for a, b in zip(before, after):
                segments += plot([a[0], b[0]], [a[1], b[1]], "-", color=color, linewidth=1.2, zorder=8)
            return segments

        if step.stage == "grow":
            artists += plot(previous[:, 0], previous[:, 1], "o:", color="white", alpha=0.75, markersize=4,
                            markeredgecolor="black", markeredgewidth=0.6, linewidth=1.2, zorder=5,
                            label="Before growth")
            artists += draw_string(images, step.reactant_count, "o-", markersize=4, linewidth=1.5, zorder=6)
            artists.append(self.axis.scatter(*step.new_node, s=90, color=GSM_COLOR, edgecolor="black",
                                             zorder=9, label="New node (on spline)"))
            message = (f"growth: node {step.addition + 1}/{step.total_additions} added at "
                       f"{np.round(step.new_node, 3)}; the other nodes are reparametrized")
        elif step.stage == "evolve":
            artists += plot(previous[:, 0], previous[:, 1], "o:", color="white", alpha=0.75, markersize=4,
                            markeredgecolor="black", markeredgewidth=0.6, linewidth=1.2, zorder=5,
                            label="Previous string")
            artists += node_shifts(previous, images, "yellow")
            artists += draw_string(images, step.reactant_count, "o-", markersize=5, linewidth=1.5, zorder=7)
            artists += plot([], [], "o-", color=GSM_COLOR, label="Evolved string")
            moved = float(np.max(np.linalg.norm(images - previous, axis=1)))
            message = (f"{phase} — evolve: nodes move along the normal force f⊥ "
                       f"(max |f⊥| = {step.max_perpendicular_gradient:.3g}, max shift {moved:.3g})")
        else:  # reparametrize
            artists += plot(previous[:, 0], previous[:, 1], "o", color="gray", markersize=5,
                            markeredgecolor="black", markeredgewidth=0.6, zorder=6, label="Evolved string")
            artists += node_shifts(previous, images, "white")
            artists += draw_string(images, step.reactant_count, "o-", markersize=6, linewidth=1.5, zorder=7,
                                   markerfacecolor="none", markeredgewidth=1.5)
            artists += plot([], [], "o-", color=GSM_COLOR, markerfacecolor="none", label="Reparametrized")
            moved = float(np.max(np.linalg.norm(images - previous, axis=1)))
            spacing = (f"equal arc length{'' if joined else ' in each fragment, gap kept vacant'}")
            message = f"{phase} — reparametrize: nodes slide along the spline to {spacing} (max shift {moved:.3g})"

        self.frame_artists = artists
        self.frame_artists.append(self.axis.legend(loc="upper right", fontsize=8))
        self.status.set_text(f"GSM — {message}\nSo far: {format_counts(step)}")
        self.refresh(flush=False)

    # --- Final results ------------------------------------------------------
    def draw_results(self) -> None:
        self.draw_surface()
        colors = {"FSM": FSM_COLOR, "GSM": GSM_COLOR}
        summaries = []
        for result in self.results:
            color = colors[result.method]
            ts_index = int(result.metadata.get("transition_state_index", np.argmax(result.energies)))
            ts_point = result.images[ts_index]
            if not np.array_equal(ts_point, result.transition_state):
                raise RuntimeError("Internal error: transition state is not its indexed path image")
            curve = display_curve(result.images)
            self.axis.plot(
                curve[:, 0], curve[:, 1], "-", color=color,
                linewidth=2.0, label=f"{result.method} MEP", zorder=6,
            )
            self.axis.plot(
                result.images[:, 0], result.images[:, 1], "o", color=color,
                markersize=3.5, linestyle="None", zorder=7,
            )
            self.axis.scatter(
                *ts_point, marker="*", s=230, facecolors="none", linewidths=2.0,
                edgecolors="black", label=f"{result.method} TS (path node {ts_index})", zorder=9,
            )
            # Preserve the colored path node at the center of the hollow star.
            self.axis.scatter(*ts_point, marker="o", s=18, color=color, zorder=10)
            ts_path_distance = float(np.linalg.norm(ts_point - result.images[ts_index]))
            optimizer = result.metadata.get("node_optimizer") or result.metadata.get("string_optimizer")
            summaries.append(
                f"{result.method} [{optimizer}; TS: {result.metadata['ts_refinement']}]: TS image {ts_index}={np.round(ts_point, 4)}, "
                f"E={result.transition_state_energy:.4f}, barrier={result.barrier_from_reactant:.4f}, "
                f"node distance={ts_path_distance:.1e}\n"
                f"    {format_counts(result)}"
            )
        self.axis.legend(loc="upper right", fontsize=8)
        self.status.set_text("\n".join(summaries))
        self.run_button.label.set_text("Run")
        self.refresh()


def _install_debug_log(gui: StringMethodGUI) -> None:  # TEMP-DEBUG
    import faulthandler, os, signal
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_debug.log")
    out = open(path, "w", buffering=1)
    faulthandler.register(signal.SIGUSR1, file=out, all_threads=True)
    canvas = gui.figure.canvas
    t0 = time.time()

    def log(tag: str, event=None) -> None:
        where = ""
        if event is not None and hasattr(event, "x"):
            where = f" b={getattr(event, 'button', None)} xy=({event.x:.0f},{event.y:.0f}) in={'none' if event.inaxes is None else type(event.inaxes).__name__}"
        out.write(
            f"t={time.time() - t0:7.2f} {tag}{where} | drag={gui.delay_slider.drag_active} "
            f"grab={canvas.mouse_grabber is not None} lock={canvas.widgetlock.locked()} "
            f"pending={getattr(canvas, '_draw_pending', None)} drawing={getattr(canvas, '_is_drawing', None)} "
            f"timer={'on' if gui.timer else 'off'} frame={gui.frame_index}/{len(gui.frames)} delay={gui.delay_slider.val}\n"
        )

    for name in ("button_press_event", "button_release_event", "draw_event", "figure_leave_event", "figure_enter_event"):
        canvas.mpl_connect(name, lambda e, n=name: log(n, e))
    last = [0.0]

    def motion(e) -> None:
        if time.time() - last[0] > 0.25:
            last[0] = time.time()
            log("motion", e)

    canvas.mpl_connect("motion_notify_event", motion)
    log("ready")


def main() -> None:
    gui = StringMethodGUI()
    _install_debug_log(gui)  # TEMP-DEBUG
    plt.show()


if __name__ == "__main__":
    main()
