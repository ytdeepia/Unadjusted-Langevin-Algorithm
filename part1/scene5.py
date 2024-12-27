from manim import *

import numpy as np


class Scene1_5(Scene):
    def construct(self):

        # Scatter points into probability distribution
        self.next_section(skip_animations=False)

        # Generate data from two Gaussian distributions
        n_points = 150
        mean1 = np.array([6.5, -2.5])
        cov1 = np.array([[4.0, 2.0], [2.0, 4.0]]) * 1.1

        mean2 = np.array([-6, 3])
        cov2 = np.array([[4.0, -1.6], [-1.6, 4.0]]) * 1.1

        points1 = np.random.multivariate_normal(mean1, cov1, n_points)
        points2 = np.random.multivariate_normal(mean2, cov2, n_points)

        xmin = -18
        xmax = 18
        ymin = -11
        ymax = 12

        x_span = xmax - xmin  # 32
        y_span = ymax - ymin  # 24
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

        image_plane = Rectangle(color=WHITE, height=6, width=10)
        image_plane_txt = Tex("Image plane", color=WHITE).next_to(
            image_plane, UP, buff=0.1
        )

        dot1 = Dot(
            point=axes.c2p(mean1[0], mean1[1]), radius=0.075, color=BLUE
        ).set_z_index(1)
        dot2 = Dot(
            point=axes.c2p(mean2[0], mean2[1]), radius=0.075, color=BLUE
        ).set_z_index(1)

        label_1 = (
            MathTex(r"x_1", color=BLUE).next_to(dot1, UP, buff=0.12).set_z_index(1)
        )
        label_2 = (
            MathTex(r"x_2", color=BLUE).next_to(dot2, UP, buff=0.12).set_z_index(1)
        )

        img1 = (
            ImageMobject("img/original_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(2)
        )
        img1.add(SurroundingRectangle(img1, color=WHITE, buff=0))
        img1.next_to(dot1, UR, buff=1)
        line1 = Line(dot1, img1, color=WHITE)

        img2 = (
            ImageMobject("img/original_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(2)
        )
        img2.add(SurroundingRectangle(img2, color=WHITE, buff=0))
        img2.next_to(dot2, UL, buff=1)
        line2 = Line(dot2, img2, color=WHITE)

        self.play(Create(image_plane), Write(image_plane_txt))

        self.play(Create(dot1), Write(label_1))
        self.play(Create(dot2), Write(label_2))

        self.wait(0.6)

        self.play(FadeOut(image_plane, image_plane_txt))

        self.play(LaggedStart(Create(line1), FadeIn(img1), lag_ratio=0.5))
        self.play(LaggedStart(Create(line2), FadeIn(img2), lag_ratio=0.5))

        self.wait(0.5)

        self.play(FadeOut(img1, line1, img2, line2))

        self.play(
            LaggedStartMap(GrowFromPoint, noisy_dots1, point=dot1.get_center()),
            LaggedStartMap(GrowFromPoint, noisy_dots2, point=dot2.get_center()),
            run_time=3,
        )

        noisy_points_label = MathTex(
            r"\mathbf{y}_n", r"\sim p(", r"\mathbf{y}", r")"
        ).next_to(noisy_dots1, UP, buff=0.5)
        noisy_points_label[0].set_color(ORANGE)
        noisy_points_label[2].set_color(ORANGE)

        self.play(Write(noisy_points_label))

        self.wait(0.7)

        noisy_img1 = (
            ImageMobject("img/noisy_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(2)
        )
        noisy_img1.add(SurroundingRectangle(noisy_img1, color=WHITE, buff=0))
        noisy_img1.next_to(noisy_dots2, DL, buff=-1.5)
        noisy_line1 = Line(noisy_dots1, noisy_img1, color=WHITE)

        self.play(LaggedStart(Create(noisy_line1), FadeIn(noisy_img1), lag_ratio=0.5))

        self.wait(3)
        self.play(
            LaggedStart(
                FadeOut(
                    noisy_dots1,
                    noisy_dots2,
                    dot1,
                    dot2,
                    label_1,
                    label_2,
                    noisy_points_label,
                    noisy_line1,
                    noisy_img1,
                ),
                AnimationGroup(Create(ellipses1), Create(ellipses2)),
                lag_ratio=0.2,
            ),
            run_time=5,
        )

        self.wait(0.7)

        noisy_label = MathTex(r"p(", r"\mathbf{y}", r")").to_edge(UP)
        noisy_label[1].set_color(ORANGE)

        self.play(Write(noisy_label))

        self.wait(0.7)

        # Probability distribution to score
        self.next_section(skip_animations=False)

        # Create vector field for the gradients
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

            # Compute gradients of individual Gaussians
            diff1 = pos - mean1
            diff2 = pos - mean2
            inv_cov1 = np.linalg.inv(cov1)
            inv_cov2 = np.linalg.inv(cov2)
            grad1 = -np.dot(inv_cov1, diff1)
            grad2 = -np.dot(inv_cov2, diff2)

            # Score is gradient of log probability = (∇p) / p
            numerator = w1 * p1 * grad1 + w2 * p2 * grad2
            denominator = w1 * p1 + w2 * p2
            score = numerator / denominator

            return np.array([score[0], score[1], 0])

        vector_field = ArrowVectorField(
            func=lambda pos: gradient_field(pos[0], pos[1]),
            x_range=[xmin, xmax, 0.5],
            y_range=[ymin, ymax, 0.5],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            colors=[BLUE_D, YELLOW, RED_E],
        )
        vector_field.fit_to_coordinate_system(axes)

        self.wait(0.7)

        self.play(
            LaggedStart(
                FadeOut(ellipses1, ellipses2, noisy_label),
                Create(vector_field),
                lag_ratio=0.3,
            ),
            run_time=4,
        )
        score_label = MathTex(r"\nabla \log p(", r"\mathbf{y}", r")").to_edge(UP)
        score_label[1].set_color(ORANGE)

        self.play(Write(score_label))

        self.wait(0.8)

        # Applying the denoiser
        self.next_section(skip_animations=False)

        self.play(FadeOut(vector_field, score_label), run_time=1)

        tweedie_formula = (
            MathTex(
                r"s_{\theta}",
                r"(",
                r"\mathbf{y}",
                r") = ",
                r"\mathbf{y}",
                r" + \sigma^2 \nabla \log p(",
                r"\mathbf{y}",
                r")",
            )
            .scale(0.8)
            .to_edge(UP)
            .set_z_index(2)
        )
        tweedie_formula[0].set_color(TEAL)
        tweedie_formula[2].set_color(ORANGE)
        tweedie_formula[4].set_color(ORANGE)
        tweedie_formula[6].set_color(ORANGE)

        tweedie_rect = SurroundingRectangle(
            tweedie_formula, color=WHITE, buff=0.1
        ).set_z_index(2)

        self.play(
            LaggedStart(Write(tweedie_formula), Create(tweedie_rect), lag_ratio=0.3),
            run_time=2,
        )

        self.play(FadeIn(ellipses1), FadeIn(ellipses2))

        self.wait(0.7)

        moving_point = Dot(
            point=axes.c2p(0, 0), radius=0.075, color=ORANGE
        ).set_z_index(1)
        moving_label = (
            MathTex(r"\mathbf{y}", color=ORANGE)
            .next_to(moving_point, UP, buff=0.1)
            .set_z_index(1)
        )
        moving_label.add_updater(lambda m: m.next_to(moving_point, UP, buff=0.1))

        self.play(Create(moving_point), Write(moving_label))

        scale_arrow = 0.4

        score_arrow = Arrow(
            start=moving_point.get_center(),
            end=moving_point.get_center() + gradient_field(0, 0) * scale_arrow,
            color=TEAL,
            buff=0,
            max_tip_length_to_length_ratio=0.3,
        )
        self.play(Create(score_arrow))

        self.wait(2)

        self.play(moving_point.animate.move_to(score_arrow.get_end()))

        self.play(moving_point.animate.move_to(score_arrow.get_start()))

        self.wait(0.8)

        def update_arrow(arrow):
            pos = axes.p2c(moving_point.get_center())
            score = gradient_field(pos[0], pos[1])
            arrow.put_start_and_end_on(
                moving_point.get_center(),
                moving_point.get_center() + score * scale_arrow,
            )

        score_arrow.add_updater(update_arrow)

        about_point1 = axes.c2p(mean1[0], mean1[1], 0)
        about_point2 = axes.c2p(mean2[0], mean2[1], 0)

        self.play(
            Rotating(
                moving_point,
                radians=2 * PI,
                about_point=about_point1,
                rate_func=smooth,
                run_time=7,
            )
        )

        self.wait(0.7)

        self.play(
            Rotating(
                moving_point,
                radians=2 * PI,
                about_point=about_point2,
                rate_func=smooth,
                run_time=5,
            )
        )

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_5()
    scene.render()
