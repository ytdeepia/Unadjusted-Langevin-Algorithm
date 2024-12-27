from manim import *

import numpy as np


class Scene2_1(Scene):
    def construct(self):
        # Intuitive explanation
        self.next_section(skip_animations=False)

        np.random.seed(0)

        n_points = 300
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

        x_span = xmax - xmin
        y_span = ymax - ymin  #
        base_length = 12

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
        n_ellipses = 10
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

        noisy_dots1.set_z_index(1)
        noisy_dots2.set_z_index(1)

        self.play(LaggedStartMap(Create, VGroup(noisy_dots1, noisy_dots2), run_time=4))

        self.wait(0.7)

        def gaussian_pdf(x, mean, cov):
            diff = x - mean
            return np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff) / np.sqrt(
                (2 * np.pi) ** 2 * np.linalg.det(cov)
            )

        def gradient_field(x, y):
            pos = np.array([x, y])
            w1, w2 = 0.5, 0.5

            p1 = gaussian_pdf(pos, mean1, cov1)
            p2 = gaussian_pdf(pos, mean2, cov2)

            diff1 = pos - mean1
            diff2 = pos - mean2
            inv_cov1 = np.linalg.inv(cov1)
            inv_cov2 = np.linalg.inv(cov2)
            grad1 = -np.dot(inv_cov1, diff1)
            grad2 = -np.dot(inv_cov2, diff2)

            numerator = w1 * p1 * grad1 + w2 * p2 * grad2
            denominator = w1 * p1 + w2 * p2
            score = numerator / denominator

            return np.array([score[0], score[1], 0])

        self.play(Create(ellipses1))
        self.play(Create(ellipses2))

        self.wait(0.6)

        vector_field = ArrowVectorField(
            func=lambda pos: gradient_field(pos[0], pos[1]),
            x_range=[xmin, xmax, 0.5],
            y_range=[ymin, ymax, 0.5],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            colors=[BLUE_D, YELLOW, RED_E],
        )
        vector_field.fit_to_coordinate_system(axes)

        self.play(Create(vector_field), run_time=2)

        self.wait(0.8)

        self.play(FadeOut(vector_field, noisy_dots1, noisy_dots2), run_time=0.7)
        gradient_step = 0.2

        moving_dot = Dot(point=axes.c2p(4, 8), color=BLUE)

        self.play(Create(moving_dot))

        score_arrow = Arrow(
            start=moving_dot.get_center(),
            end=moving_dot.get_center() + gradient_field(0, 0) * gradient_step,
            color=TEAL,
            buff=0,
            max_tip_length_to_length_ratio=0.3,
        )

        def update_arrow(arrow):
            pos = axes.p2c(moving_dot.get_center())
            score = gradient_field(pos[0], pos[1])
            arrow.put_start_and_end_on(
                moving_dot.get_center(),
                moving_dot.get_center() + score * gradient_step,
            )

        score_arrow.add_updater(update_arrow)

        self.play(Create(score_arrow))

        self.play(moving_dot.animate.move_to(axes.c2p(3, 4)), run_time=1.5)
        self.play(moving_dot.animate.move_to(axes.c2p(1, 8)), run_time=1.5)
        self.play(moving_dot.animate.move_to(axes.c2p(4, 8)), run_time=1.5)

        self.play(FadeOut(score_arrow), run_time=0.7)

        curr_coord = axes.p2c(moving_dot.get_center())

        arrows = VGroup()

        for _ in range(10):
            noise = np.random.normal(0, 1, 2)
            noise = np.array([noise[0], noise[1], 0])
            # Create arrow based on current position and gradient
            moving_arrow = Arrow(
                start=moving_dot.get_center(),
                end=moving_dot.get_center()
                + gradient_field(curr_coord[0], curr_coord[1]) * gradient_step,
                color=TEAL,
                buff=0,
                max_tip_length_to_length_ratio=0.3,
            )

            arrows.add(moving_arrow)

            self.play(Create(moving_arrow), run_time=0.5)
            self.play(moving_dot.animate.move_to(moving_arrow.get_end()), run_time=0.5)

            curr_coord = axes.p2c(moving_dot.get_center())

        self.wait(0.7)

        langevin_formula_1 = (
            MathTex(
                r"x_{t+1} = x_t + \eta ",
                r"\nabla \log p(x_t)",
            )
            .scale(0.6)
            .to_corner(UR, buff=1)
        )

        self.play(Write(langevin_formula_1))

        self.wait(0.7)

        rect_grad = SurroundingRectangle(
            langevin_formula_1[1], buff=0.1, color=WHITE
        ).set_z_index(1)
        self.play(Create(rect_grad))
        self.wait(2)
        self.play(FadeOut(rect_grad))

        langevin_formula_2 = (
            MathTex(
                r"x_{t+1} = x_t + \frac{\eta}{\sigma^2} (",
                r"s_{\theta}",
                r"(x_t) - x_t)",
            )
            .scale(0.6)
            .to_corner(UR, buff=1)
        )
        langevin_formula_2[1].set_color(TEAL)

        self.play(Transform(langevin_formula_1, langevin_formula_2))

        self.wait(0.9)

        self.play(
            FadeOut(arrows), moving_dot.animate.move_to(axes.c2p(-6, -4)), run_time=0.7
        )

        moving_dot_label = (
            MathTex("x_t").scale(0.5).next_to(moving_dot, UR, buff=0.1).set_z_index(3)
        )
        self.play(Write(moving_dot_label))

        moving_dot_label.add_updater(lambda m: m.next_to(moving_dot, UR, buff=0.1))

        arrows = VGroup()

        for _ in range(10):
            noise = np.random.normal(0, 1, 2)
            noise = np.array([noise[0], noise[1], 0])

            moving_arrow = Arrow(
                start=moving_dot.get_center(),
                end=moving_dot.get_center()
                + gradient_field(curr_coord[0], curr_coord[1]) * gradient_step,
                color=TEAL,
                buff=0,
                max_tip_length_to_length_ratio=0.3,
            )

            arrows.add(moving_arrow)

            self.play(Create(moving_arrow), run_time=0.5)
            self.play(moving_dot.animate.move_to(moving_arrow.get_end()), run_time=0.5)

            curr_coord = axes.p2c(moving_dot.get_center())

        self.play(
            FadeOut(arrows, langevin_formula_1),
            moving_dot.animate.move_to(axes.c2p(4, 8)),
        )

        langevin_formula_3 = (
            MathTex(
                r"x_{t+1} = x_t + \frac{\eta}{\sigma^2} (",
                r"s_{\theta}",
                r"(x_t) - x_t) + ",
                r"\sqrt{2\eta} z",
            )
            .scale(0.7)
            .next_to(ellipses1, direction=DOWN, buff=0.5)
            .set_z_index(1)
        )
        langevin_formula_3[1].set_color(TEAL)
        z_label = (
            MathTex(r"z \sim \mathcal{N}(0, I)")
            .scale(0.8)
            .next_to(langevin_formula_3, DOWN, buff=1)
        )

        self.play(Write(langevin_formula_3))

        self.wait(0.6)

        brace_noise = Brace(langevin_formula_3[3], DOWN, buff=0.1)
        brace_noise_text = brace_noise.get_text("Small perturbation").scale(0.6)
        self.play(
            LaggedStart(Create(brace_noise), Write(brace_noise_text), lag_ratio=0.3),
            run_time=1.5,
        )

        self.play(FadeOut(brace_noise, brace_noise_text))
        rect2 = SurroundingRectangle(
            langevin_formula_3, buff=0.1, color=WHITE
        ).set_z_index(1)
        self.play(Create(rect2))
        self.play(Write(z_label))

        self.wait(0.75)

        temp = 0.1

        arrows = VGroup()

        for _ in range(10):
            noise = np.random.normal(0, 1, 2)
            noise = np.array([noise[0], noise[1], 0])

            moving_arrow = Arrow(
                start=moving_dot.get_center(),
                end=moving_dot.get_center()
                + gradient_field(curr_coord[0], curr_coord[1]) * gradient_step
                + ((2 * gradient_step) ** 0.5) * noise,
                color=TEAL,
                buff=0,
                max_tip_length_to_length_ratio=0.3,
            )

            arrows.add(moving_arrow)

            self.play(Create(moving_arrow), run_time=0.5)
            self.play(moving_dot.animate.move_to(moving_arrow.get_end()), run_time=0.5)

            curr_coord = axes.p2c(moving_dot.get_center())

        self.wait(0.8)

        # Example with a a lot of points
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(arrows, moving_dot, z_label, moving_dot_label),
            VGroup(langevin_formula_3, rect2).animate.to_corner(UR, buff=1),
        )

        dots = VGroup()
        n_starting_points = 600
        random_points = np.random.uniform(
            low=[xmin, ymin], high=[xmax, ymax], size=(n_starting_points, 2)
        )
        for point in random_points:
            dots.add(Dot(point=axes.c2p(point[0], point[1]), color=BLUE, radius=0.025))

        self.play(FadeIn(dots))

        self.wait(0.8)

        temp = 0.5

        def update_dot(dot, dt):
            x, y = axes.p2c(dot.get_center())
            noise = np.random.normal(0, 1, 2)
            noise = np.array([noise[0], noise[1], 0])

            new_pos = (
                dot.get_center()
                + 0.3 * dt * gradient_field(x, y)
                + temp * np.sqrt(2 * 0.3 * dt) * noise
            )
            dot.move_to(new_pos)

        for dot in dots:
            dot.add_updater(update_dot)

        dots.clear_updaters()

        self.wait(0.6)

        self.play(FadeIn(noisy_dots1, noisy_dots2))

        self.wait(0.7)

        self.play(FadeOut(*self.mobjects, shift=0.5 * RIGHT))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_1()
    scene.render()
