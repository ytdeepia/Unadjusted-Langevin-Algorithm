from manim import *

import numpy as np


class Scene2_2(Scene):
    def construct(self):
        # Training code
        self.next_section(skip_animations=False)

        code = Code(
            file_name="./minimal_training.py",
            background="window",
            language="python",
            insert_line_no=False,
            style=Code.styles_list[15],
        ).scale_to_fit_height(config.frame_height - 1)

        self.play(Create(code.background_mobject))

        self.wait(0.5)

        self.play(Write(code.code[:3]), run_time=2)

        self.wait(0.7)
        self.play(Write(code.code[3:15]))

        self.wait(0.7)

        self.play(Indicate(code.code[11], color=WHITE), run_time=1.5)

        self.wait(0.6)

        self.play(Write(code.code[15:21]))

        self.wait(0.7)

        self.play(Write(code.code[21:25]))

        self.wait(0.6)
        self.play(Write(code.code[25:]), run_time=5)

        self.wait(0.5)

        self.play(Indicate(code.code[35:37], color=WHITE), run_time=1.5)

        self.play(code.animate.scale(0.6).to_corner(UL))

        # Denoiser training
        self.next_section(skip_animations=False)

        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * np.exp(-0.6 * x) + 0.3 * np.exp(-0.2 * x) + 0.1

        noise = np.random.normal(0, 0.02, x.shape)
        y = y + noise

        axes = (
            Axes(
                x_range=[0, 10],
                y_range=[0, 2.5],
                x_axis_config={"include_numbers": True},
                y_axis_config={"include_numbers": False},
                tips=False,
            )
            .scale(0.5)
            .to_corner(UR, buff=0.75)
        )

        curve = axes.plot_line_graph(
            x_values=x,
            y_values=y,
            line_color=BLUE,
            add_vertex_dots=False,
        )

        label_x, label_y = axes.get_axis_labels(x_label="Epochs", y_label="MSE")
        label_x.scale(0.5).shift(0.9 * DOWN)
        label_y.scale(0.5).shift(LEFT + 0.25 * UP)

        self.play(Create(axes))
        self.play(Write(label_x), Write(label_y))
        self.play(Create(curve), run_time=4)

        self.wait(0.7)

        noisy_0 = ImageMobject("./img/noisy_0.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )
        denoised_0 = ImageMobject("./img/denoised_0.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )
        noisy_1 = ImageMobject("./img/noisy_1.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )
        denoised_1 = ImageMobject("./img/denoised_1.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )
        noisy_2 = ImageMobject("./img/noisy_2.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )
        denoised_2 = ImageMobject("./img/denoised_2.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )
        noisy_3 = ImageMobject("./img/noisy_3.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )
        denoised_3 = ImageMobject("./img/denoised_3.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )
        noisy_4 = ImageMobject("./img/noisy_4.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )
        denoised_4 = ImageMobject("./img/denoised_4.png").set_resampling_algorithm(
            RESAMPLING_ALGORITHMS["none"]
        )

        imgs_noisy = Group(
            noisy_0,
            noisy_1,
            noisy_2,
            noisy_3,
            noisy_4,
        )

        imgs_denoised = Group(
            denoised_0,
            denoised_1,
            denoised_2,
            denoised_3,
            denoised_4,
        )

        for img in imgs_noisy:
            img.add(SurroundingRectangle(img, buff=0.0, color=WHITE))
        for img in imgs_denoised:
            img.add(SurroundingRectangle(img, buff=0.0, color=WHITE))

        imgs_noisy.arrange(direction=RIGHT, buff=0.1)
        imgs_noisy.scale_to_fit_width(axes.width).next_to(axes, DOWN, buff=0.75)
        imgs_denoised.arrange(direction=RIGHT, buff=0.1)
        imgs_denoised.scale_to_fit_width(axes.width).next_to(
            imgs_noisy, DOWN, buff=0.25
        )
        noisy_label = (
            Tex("Noisy Images").scale(0.6).next_to(imgs_noisy, LEFT, buff=0.25)
        )
        denoised_label = (
            Tex("Denoised Images").scale(0.6).next_to(imgs_denoised, LEFT, buff=0.25)
        )

        self.wait(0.6)

        self.play(FadeIn(noisy_label))
        self.play(LaggedStartMap(FadeIn, imgs_noisy, lag_ratio=0.3), run_time=2.5)
        self.play(FadeIn(denoised_label))
        self.play(LaggedStartMap(FadeIn, imgs_denoised, lag_ratio=0.3), run_time=2.5)

        self.wait(0.3)

        self.play(FadeOut(*self.mobjects))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_2()
    scene.render()
