from manim import *


class Scene2_3(Scene):
    def construct(self):

        # Langevin sampling code
        self.next_section(skip_animations=False)

        code = Code(
            file_name="./minimal_sampling.py",
            background="window",
            language="python",
            insert_line_no=False,
            style=Code.styles_list[15],
        )
        code.scale_to_fit_width(0.5 * config.frame_width)

        code.to_edge(LEFT, buff=1)

        self.play(Create(code.background_mobject))

        self.wait(0.6)

        self.play(Write(code.code[:3]), run_time=2)

        self.wait(0.6)

        self.play(Write(code.code[3:6]), run_time=2)
        noising_formula = MathTex(
            r"\mathbf{y}", r"= \mathbf{x} + \sigma \mathbf{\epsilon}"
        )
        noising_formula[0].set_color(ORANGE)
        noising_formula.font_size = 24

        noising_formula.next_to(code.code[4], RIGHT, buff=3)

        self.play(Write(noising_formula))

        self.wait(0.7)

        self.play(Write(code.code[6:9]))

        grad_approximation_formula = MathTex(
            r"\nabla \log p(",
            r"\mathbf{y}",
            r") = \frac{1}{\sigma^2} (s_{\theta}(",
            r"\mathbf{y}",
            r") - ",
            r"\mathbf{y}",
            r")",
        )
        grad_approximation_formula[1].set_color(ORANGE)
        grad_approximation_formula[3].set_color(ORANGE)
        grad_approximation_formula[5].set_color(ORANGE)
        grad_approximation_formula.font_size = 24

        grad_approximation_formula.next_to(code.code[8], RIGHT, buff=2)

        self.play(Write(grad_approximation_formula))

        self.wait(0.7)

        self.play(Write(code.code[9:]))

        langevin_update = MathTex(
            r"\mathbf{x}_{t+1}",
            r" = \mathbf{x}_{t} + \eta \nabla \log p(",
            r"\mathbf{y}",
            r") + \sqrt{2 \eta }\mathbf{\epsilon}",
        )
        langevin_update[2].set_color(ORANGE)
        langevin_update.font_size = 24

        langevin_update.next_to(code.code[12], RIGHT, buff=1.5)

        self.play(Write(langevin_update))

        self.wait(0.6)

        self.play(FadeOut(*self.mobjects, shift=0.5 * RIGHT), run_time=0.7)

        # Langevin samples from noisy inputs
        self.next_section(skip_animations=False)

        original_sample = (
            ImageMobject("./img/denoised_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(1.5)
        )
        original_sample.add(
            SurroundingRectangle(original_sample, buff=0.0, color=WHITE)
        )
        # Create 8 sample groups, each containing images for iterations 0-2000
        samples = []
        for sample_idx in range(8):
            sample_group = Group()
            for t in range(0, 2000, 50):
                img = ImageMobject(
                    f"./img/sample_{sample_idx}_iter_{t}.png"
                ).set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                sample_group.add(img)
            samples.append(sample_group)

        # Unpack into individual variables
        (
            sample_1,
            sample_2,
            sample_3,
            sample_4,
            sample_5,
            sample_6,
            sample_7,
            sample_8,
        ) = samples

        original_label = Tex(r"Original Image").scale(0.6).to_edge(UP, buff=0.25)
        original_label.add(Underline(original_label))
        original_sample.next_to(original_label, DOWN, buff=0.25)
        samples_label = (
            Tex(r"Langevin Samples").scale(0.6).next_to(original_sample, DOWN, buff=0.5)
        )
        samples_label.add(Underline(samples_label))

        samples = (
            Group(
                sample_1,
                sample_2,
                sample_3,
                sample_4,
                sample_5,
                sample_6,
                sample_7,
                sample_8,
            )
            .arrange_in_grid(rows=2, cols=4, buff=0.05)
            .scale_to_fit_height(0.4 * config.frame_height)
            .next_to(samples_label, DOWN, buff=0.5)
        )
        rects = VGroup(
            *[
                SurroundingRectangle(sample[0], buff=0.0, color=WHITE, z_index=2)
                for sample in [
                    sample_1,
                    sample_2,
                    sample_3,
                    sample_4,
                    sample_5,
                    sample_6,
                    sample_7,
                    sample_8,
                ]
            ]
        )

        self.play(
            LaggedStart(
                Create(original_label),
                FadeIn(
                    original_sample,
                ),
            )
        )

        iter_counter = (
            Tex("Iteration: 0").scale(0.7).next_to(original_sample, LEFT, buff=1.5)
        )

        self.play(
            FadeIn(
                sample_1[0],
                sample_2[0],
                sample_3[0],
                sample_4[0],
                sample_5[0],
                sample_6[0],
                sample_7[0],
                sample_8[0],
                samples_label,
                rects,
                iter_counter,
            )
        )

        self.wait(0.6)

        for idx in range(len(sample_1)):
            self.add(sample_1[idx])
            self.add(sample_2[idx])
            self.add(sample_3[idx])
            self.add(sample_4[idx])
            self.add(sample_5[idx])
            self.add(sample_6[idx])
            self.add(sample_7[idx])
            self.add(sample_8[idx])

            self.remove(sample_1[idx - 1])
            self.remove(sample_2[idx - 1])
            self.remove(sample_3[idx - 1])
            self.remove(sample_4[idx - 1])
            self.remove(sample_5[idx - 1])
            self.remove(sample_6[idx - 1])
            self.remove(sample_7[idx - 1])
            self.remove(sample_8[idx - 1])
            self.remove(iter_counter)
            iter_counter = (
                Tex(f"Iteration: {idx * 50}")
                .scale(0.7)
                .next_to(original_sample, LEFT, buff=1.5)
            )
            self.add(iter_counter)
            self.wait(0.2)

        self.wait(0.6)

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))

        # Langevin from pure noise
        self.next_section(skip_animations=False)

        samples = []
        for i in range(16):
            group = Group()
            for t in range(0, 3000, 50):
                img = ImageMobject(
                    f"./img/noisy_sample_{i}_iter_{t}.png"
                ).set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                group.add(img)
            samples.append(group)

        (
            sample_1,
            sample_2,
            sample_3,
            sample_4,
            sample_5,
            sample_6,
            sample_7,
            sample_8,
            sample_9,
            sample_10,
            sample_11,
            sample_12,
            sample_13,
            sample_14,
            sample_15,
            sample_16,
        ) = samples

        samples_label = (
            Tex(r"Langevin Samples from noise").scale(0.6).to_edge(UP, buff=0.25)
        )
        samples_label.add(Underline(samples_label))

        iter_counter = (
            Tex("Iteration: 0").scale(0.7).next_to(samples_label, LEFT, buff=1.5)
        )

        samples_g = (
            Group(
                sample_1,
                sample_2,
                sample_3,
                sample_4,
                sample_5,
                sample_6,
                sample_7,
                sample_8,
                sample_9,
                sample_10,
                sample_11,
                sample_12,
                sample_13,
                sample_14,
                sample_15,
                sample_16,
            )
            .arrange_in_grid(rows=4, cols=4, buff=0.05)
            .scale_to_fit_height(0.7 * config.frame_height)
            .next_to(samples_label, DOWN, buff=0.5)
        )

        rects = VGroup(
            *[
                SurroundingRectangle(sample[0], buff=0.0, color=WHITE, z_index=2)
                for sample in samples
            ]
        )

        self.play(
            FadeIn(samples_label),
            FadeIn(
                sample_1[0],
                sample_2[0],
                sample_3[0],
                sample_4[0],
                sample_5[0],
                sample_6[0],
                sample_7[0],
                sample_8[0],
                sample_9[0],
                sample_10[0],
                sample_11[0],
                sample_12[0],
                sample_13[0],
                sample_14[0],
                sample_15[0],
                sample_16[0],
                rects,
            ),
        )

        self.wait(0.6)

        for idx in range(len(sample_1)):
            self.add(sample_1[idx])
            self.add(sample_2[idx])
            self.add(sample_3[idx])
            self.add(sample_4[idx])
            self.add(sample_5[idx])
            self.add(sample_6[idx])
            self.add(sample_7[idx])
            self.add(sample_8[idx])
            self.add(sample_9[idx])
            self.add(sample_10[idx])
            self.add(sample_11[idx])
            self.add(sample_12[idx])
            self.add(sample_13[idx])
            self.add(sample_14[idx])
            self.add(sample_15[idx])
            self.add(sample_16[idx])

            self.remove(sample_1[idx - 1])
            self.remove(sample_2[idx - 1])
            self.remove(sample_3[idx - 1])
            self.remove(sample_4[idx - 1])
            self.remove(sample_5[idx - 1])
            self.remove(sample_6[idx - 1])
            self.remove(sample_7[idx - 1])
            self.remove(sample_8[idx - 1])
            self.remove(sample_9[idx - 1])
            self.remove(sample_10[idx - 1])
            self.remove(sample_11[idx - 1])
            self.remove(sample_12[idx - 1])
            self.remove(sample_13[idx - 1])
            self.remove(sample_14[idx - 1])
            self.remove(sample_15[idx - 1])
            self.remove(sample_16[idx - 1])
            self.remove(iter_counter)
            iter_counter = (
                Tex(f"Iteration: {idx * 50}")
                .scale(0.7)
                .next_to(samples_label, LEFT, buff=1.5)
            )
            self.add(iter_counter)
            self.wait(0.2)

        self.wait(0.8)

        self.play(FadeOut(*self.mobjects))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_3()
    scene.render()
