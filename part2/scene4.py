from manim import *

import numpy as np


class Scene2_4(Scene):
    def construct(self):
        np.random.seed(0)

        # Generate data from two Gaussian distributions
        n_points = 150
        mean1 = np.array([-8, 6.5])
        cov = np.array([[3.0, 0.0], [0.0, 3.0]]) * 1.2

        mean2 = np.array([8, -3.5])
        cov1 = cov
        cov2 = cov

        points1 = np.random.multivariate_normal(mean1, cov1, n_points)
        points2 = np.random.multivariate_normal(mean2, cov2, n_points)

        xmin = -18
        xmax = 18
        ymin = -11
        ymax = 12

        x_span = xmax - xmin  # 32
        y_span = ymax - ymin  # 24
        base_length = 12

        # Showing low density regions
        self.next_section(skip_animations=False)

        denoising_score_matching = MathTex(
            r"\mathbb{E}_{",
            r"\mathbf{y}",
            r"\sim p(",
            r"\mathbf{y}",
            r"), \mathbf{x} \sim p(\mathbf{x})} \left[ \|",
            r"s_{\theta}",
            r"(",
            r"\mathbf{y}",
            r") - \mathbf{x} \|^2 \right]",
        ).scale(0.8)
        denoising_score_matching[1].set_color(ORANGE)
        denoising_score_matching[3].set_color(ORANGE)
        denoising_score_matching[5].set_color(TEAL)
        denoising_score_matching[7].set_color(ORANGE)

        rect = SurroundingRectangle(denoising_score_matching, buff=0.2, color=WHITE)

        denoising_score_matching_label = (
            Tex("Denoising score matching").scale(0.8).to_edge(UP)
        )
        denoising_score_matching_label.add(Underline(denoising_score_matching_label))

        self.play(Create(denoising_score_matching_label))

        self.wait(0.4)

        self.play(
            LaggedStart(Write(denoising_score_matching), Create(rect), lag_ratio=0.3),
            run_time=2,
        )

        self.wait(0.7)

        self.play(FadeOut(*self.mobjects, shift=0.5 * RIGHT), run_time=0.8)

        axes = Axes(
            x_range=[xmin, xmax, 0.5],
            y_range=[ymin, ymax, 0.5],
            x_length=base_length,
            y_length=base_length * (y_span / x_span),  # Proportional to coordinate span
            axis_config={"color": WHITE},
        )

        noisy_dots1 = VGroup(
            *[Dot(point=axes.c2p(x, y), radius=0.025, color=ORANGE) for x, y in points1]
        )
        noisy_dots2 = VGroup(
            *[Dot(point=axes.c2p(x, y), radius=0.025, color=ORANGE) for x, y in points2]
        )

        eigenvalues1, eigenvectors1 = np.linalg.eig(cov1)
        angle1 = np.arctan2(eigenvectors1[1, 0], eigenvectors1[0, 0])
        n_ellipses = 20
        max_scale = 0.9
        min_scale = 0.1
        ellipses1 = VGroup(
            *[
                Ellipse(
                    width=2 * np.sqrt(eigenvalues1[0]) * scale,
                    height=2 * np.sqrt(eigenvalues1[1]) * scale,
                    color=ORANGE,
                    fill_opacity=0.05,
                    stroke_opacity=0,
                )
                .rotate(angle1)
                .move_to(axes.c2p(mean1[0], mean1[1]))
                for i, scale in enumerate(
                    np.linspace(max_scale, min_scale, n_ellipses)[::-1], 1
                )
            ]
        )

        eigenvalues2, eigenvectors2 = np.linalg.eig(cov2)
        angle2 = np.arctan2(eigenvectors2[1, 0], eigenvectors2[0, 0])
        ellipses2 = VGroup(
            *[
                Ellipse(
                    width=2 * np.sqrt(eigenvalues2[0]) * scale,
                    height=2 * np.sqrt(eigenvalues2[1]) * scale,
                    color=ORANGE,
                    fill_opacity=0.05,
                    stroke_opacity=0,
                )
                .rotate(angle2)
                .move_to(axes.c2p(mean2[0], mean2[1]))
                for i, scale in enumerate(
                    np.linspace(max_scale, min_scale, n_ellipses)[::-1], 1
                )
            ]
        )

        self.play(
            LaggedStartMap(GrowFromPoint, noisy_dots1, point=axes.c2p(6.5, -2.5)),
            run_time=2,
        )
        self.play(
            LaggedStartMap(GrowFromPoint, noisy_dots2, point=axes.c2p(-6, 3)),
            run_time=2,
        )

        self.play(FadeIn(ellipses1, ellipses2))

        self.wait(0.6)

        rect_low_density1 = Rectangle(
            color=RED, fill_opacity=0.2, width=5, height=2.5
        ).move_to(axes.c2p(-7.5, -4))
        rect_low_density2 = Rectangle(
            color=RED, fill_opacity=0.2, width=5, height=2.5
        ).move_to(axes.c2p(6.5, 6))
        label_low_density1 = (
            Tex("Low density region", color=RED_B)
            .scale(0.6)
            .move_to(rect_low_density1.get_center())
        )
        label_low_density2 = (
            Tex("Low density region", color=RED_B)
            .scale(0.6)
            .move_to(rect_low_density2.get_center())
        )

        self.play(Create(rect_low_density1))
        self.play(Create(rect_low_density2))

        self.play(Write(label_low_density1), Write(label_low_density2))

        self.wait(0.6)

        self.play(
            FadeOut(
                label_low_density1,
                label_low_density2,
                rect_low_density1,
                rect_low_density2,
            )
        )

        # Random walk in low density regions
        self.next_section(skip_animations=False)

        moving_dot = Dot(point=axes.c2p(6.5, 6), radius=0.025, color=TEAL)

        temp = 0.1
        gradient_step = 0.05
        arrows = VGroup()

        for _ in range(10):
            noise = np.random.normal(0, 1, 2)
            noise = np.array([noise[0], noise[1], 0])

            moving_arrow = Arrow(
                start=moving_dot.get_center(),
                end=moving_dot.get_center() + ((2 * gradient_step) ** 0.5) * noise,
                color=TEAL,
                buff=0,
                max_tip_length_to_length_ratio=0.3,
            )

            arrows.add(moving_arrow)

            self.play(Create(moving_arrow), run_time=0.5)
            self.play(moving_dot.animate.move_to(moving_arrow.get_end()), run_time=0.5)

            curr_coord = axes.p2c(moving_dot.get_center())

        self.wait(0.8)

        self.play(FadeOut(moving_dot, arrows))

        # Smoothing the score
        self.next_section(skip_animations=False)

        n_points = 500
        mean1 = np.array([-8, 6.5])
        cov = np.array([[3.0, 0.0], [0.0, 3.0]]) * 10

        mean2 = np.array([8, -3.5])
        cov1 = cov
        cov2 = cov

        new_points1 = np.random.multivariate_normal(mean1, cov1, n_points)
        new_points2 = np.random.multivariate_normal(mean2, cov2, n_points)

        eigenvalues1, eigenvectors1 = np.linalg.eig(cov1)
        angle1 = np.arctan2(eigenvectors1[1, 0], eigenvectors1[0, 0])
        n_ellipses = 20
        max_scale = 0.9
        min_scale = 0.1
        new_ellipses1 = VGroup(
            *[
                Ellipse(
                    width=2 * np.sqrt(eigenvalues1[0]) * scale,
                    height=2 * np.sqrt(eigenvalues1[1]) * scale,
                    color=ORANGE,
                    fill_opacity=0.05,
                    stroke_opacity=0,
                )
                .rotate(angle1)
                .move_to(axes.c2p(mean1[0], mean1[1]))
                for i, scale in enumerate(
                    np.linspace(max_scale, min_scale, n_ellipses)[::-1], 1
                )
            ]
        )

        eigenvalues2, eigenvectors2 = np.linalg.eig(cov2)
        angle2 = np.arctan2(eigenvectors2[1, 0], eigenvectors2[0, 0])
        new_ellipses2 = VGroup(
            *[
                Ellipse(
                    width=2 * np.sqrt(eigenvalues2[0]) * scale,
                    height=2 * np.sqrt(eigenvalues2[1]) * scale,
                    color=ORANGE,
                    fill_opacity=0.05,
                    stroke_opacity=0,
                )
                .rotate(angle2)
                .move_to(axes.c2p(mean2[0], mean2[1]))
                for i, scale in enumerate(
                    np.linspace(max_scale, min_scale, n_ellipses)[::-1], 1
                )
            ]
        )

        new_noisy_dots1 = VGroup(
            *[
                Dot(point=axes.c2p(x, y), radius=0.025, color=ORANGE)
                for x, y in new_points1
            ]
        )
        new_noisy_dots2 = VGroup(
            *[
                Dot(point=axes.c2p(x, y), radius=0.025, color=ORANGE)
                for x, y in new_points2
            ]
        )

        self.play(
            Transform(noisy_dots1, new_noisy_dots1),
            Transform(noisy_dots2, new_noisy_dots2),
            run_time=2,
        )
        self.play(
            Transform(ellipses1, new_ellipses1),
            Transform(ellipses2, new_ellipses2),
            run_time=2,
        )

        self.wait(0.7)

        diffusion_models = Tex("Diffusion models", z_index=2).scale(1.5)
        back_rect = BackgroundRectangle(diffusion_models, buff=0.3)
        self.play(
            LaggedStart(FadeIn(back_rect), Write(diffusion_models), lag_ratio=0.5),
            run_time=2,
        )

        self.play(FadeOut(*self.mobjects))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_4()
    scene.render()
