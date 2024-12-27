from manim import *

import numpy as np
from scipy.stats import norm


class Scene1_3(Scene):
    def construct(self):

        # 1D posterior mean with Gaussian prior
        self.next_section(skip_animations=False)

        x = np.linspace(0, 12, 10000)

        prior = norm.pdf(x, 3, 0.9)

        likelihood = norm.pdf(x, 9.5, 1.2)
        posterior = prior * likelihood
        posterior = posterior / (np.sum(posterior) * (x[1] - x[0]))

        posterior_mean = np.sum(x * posterior) * (x[1] - x[0])
        posterior_map = x[np.argmax(posterior)]

        axes = (
            Axes(
                x_range=[0, 12],
                y_range=[0, 0.5],
                axis_config={"color": WHITE},
                x_axis_config={"include_numbers": False, "include_ticks": False},
                y_axis_config={"include_numbers": False, "include_ticks": False},
            )
            .scale(0.8)
            .set_z_index(2)
        )

        prior_graph = axes.plot(
            lambda x: norm.pdf(x, 3, 0.9),
            color=BLUE_D,
        )
        likelihood_graph = axes.plot(lambda x: norm.pdf(x, 9.5, 1.2), color=ORANGE)
        posterior_graph = axes.plot(
            lambda t: posterior[np.abs(x - t).argmin()], color=TEAL
        )

        prior_label = Tex("Prior", color=BLUE_D).scale(0.8).move_to(axes.c2p(1, 0.3))
        prior_formula = MathTex(r"p(\mathbf{x})", color=BLUE_D).scale(0.8)
        prior_formula.next_to(prior_label, DOWN)

        likelihood_label = (
            Tex("Likelihood", color=ORANGE).scale(0.8).move_to(axes.c2p(12, 0.4))
        )
        likelihood_formula = MathTex(r"p(\mathbf{y}|\mathbf{x})", color=ORANGE).scale(
            0.8
        )
        likelihood_formula.next_to(likelihood_label, DOWN)

        posterior_label = (
            Tex("Posterior", color=TEAL).scale(0.8).move_to(axes.c2p(7.5, 0.45))
        )
        posterior_formula = MathTex(r"p(\mathbf{x}|\mathbf{y})", color=TEAL).scale(0.8)
        posterior_formula.next_to(posterior_label, DOWN)

        self.play(Create(axes))

        self.play(
            Create(prior_graph),
        )
        self.play(Write(prior_label))
        self.play(FadeIn(prior_formula))

        self.wait(0.6)

        noisy_line = Line(
            start=axes.c2p(9.5, -0.02), end=axes.c2p(9.5, 0.02), color=ORANGE
        )
        noisy_label = (
            MathTex(r"\mathbf{y}", color=ORANGE)
            .scale(0.8)
            .next_to(noisy_line, DOWN, buff=0.1)
        )

        self.play(Create(noisy_line), FadeIn(noisy_label))

        self.play(
            Create(likelihood_graph),
        )

        self.play(Write(likelihood_label))
        self.play(FadeIn(likelihood_formula))

        self.wait(0.7)

        self.play(Create(posterior_graph), run_time=2)
        self.play(Write(posterior_label))
        self.play(FadeIn(posterior_formula))

        self.wait(0.8)

        self.wait(1.5)
        self.play(Circumscribe(prior_formula, color=BLUE_D), run_time=1.5)
        self.wait(3)
        self.play(Circumscribe(likelihood_formula, color=ORANGE), run_time=1.5)

        self.wait(0.6)

        self.wait(1.2)
        map_line = DashedLine(
            start=axes.c2p(posterior_map, 0),
            end=axes.c2p(posterior_map, 0.6),
            color=TEAL_C,
        )

        map_label = Text("MAP", color=TEAL_C).scale(0.4).next_to(map_line, UP)

        self.play(Create(map_line), Write(map_label))

        self.wait(0.7)

        self.wait(2)
        self.play(Circumscribe(map_label, color=TEAL_C))

        self.wait(0.7)

        # 1D posterior mean with a complex prior
        self.next_section(skip_animations=False)

        prior1 = norm.pdf(x, 3, 1.5)
        prior2 = norm.pdf(x, 5, 0.3)
        prior = 0.7 * prior1 + 0.3 * prior2  # Uneven mixture

        posterior = prior * likelihood
        posterior = posterior / (np.sum(posterior) * (x[1] - x[0]))

        posterior_mean = np.sum(x * posterior) * (x[1] - x[0])
        posterior_map = x[np.argmax(posterior)]

        prior_graph_2 = axes.plot(
            lambda x: 0.7 * norm.pdf(x, 3, 1.5) + 0.3 * norm.pdf(x, 5, 0.3),
            color=BLUE_D,
        )
        posterior_graph_2 = axes.plot(
            lambda t: posterior[np.abs(x - t).argmin()], color=TEAL
        )

        self.play(
            FadeOut(posterior_graph, posterior_label, posterior_formula),
            FadeOut(map_line),
            FadeOut(map_label),
        )
        self.play(
            Transform(prior_graph, prior_graph_2),
        )

        self.wait(0.6)

        self.play(Create(posterior_graph_2), run_time=2)

        self.wait(0.7)

        mean_line = DashedLine(
            start=axes.c2p(posterior_mean, 0),
            end=axes.c2p(posterior_mean, 0.35),
            color=WHITE,
        )
        mean_label = (
            Text("Posterior Mean", color=WHITE).scale(0.4).next_to(mean_line, UP)
        )

        self.play(Create(mean_line), Write(mean_label))

        self.wait(0.8)

        txt = (
            Text("The denoiser is the best estimator on average")
            .to_edge(UP, buff=0.5)
            .scale(0.7)
        )
        txt_ul = Underline(txt)

        self.play(LaggedStart(Write(txt), Create(txt_ul), lag_ratio=0.5), run_time=1.5)

        self.wait(0.8)

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_3()
    scene.render()
