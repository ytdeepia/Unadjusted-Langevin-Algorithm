from manim import *


class Scene1_6(Scene):
    def construct(self):
        # Chronological timeline
        self.next_section(skip_animations=False)

        score = MathTex(
            r"\text{Estimating} ~ \nabla \log p(x) ~ \text{is pretty hard!}",
            color=WHITE,
        ).shift(UP)
        score_ul = Underline(score, buff=0.1)

        self.play(LaggedStart(Create(score_ul), Write(score), lag_ratio=0.1))

        self.wait(0.7)

        self.play(FadeOut(score), FadeOut(score_ul), run_time=0.7)

        timeline = (
            NumberLine(
                x_range=[2000, 2025, 5],
                length=10,
                include_numbers=True,
                include_tip=True,
                tick_size=0.1,
                numbers_to_include=range(2000, 2026, 5),
                decimal_number_config={
                    "group_with_commas": False,
                    "num_decimal_places": 0,
                },
            )
            .shift(1.5 * DOWN)
            .set_z_index(1)
        )

        self.play(Create(timeline), run_time=2)

        vincent_img = ImageMobject("img/pascal_vincent.png").scale_to_fit_width(1.5)
        vincent_img.add(SurroundingRectangle(vincent_img, color=WHITE, buff=0))
        vincent_label = Tex("Pascal Vincent", color=WHITE).scale(0.7)
        connection_img = ImageMobject("img/connection_score.png").scale_to_fit_width(2)
        connection_img.move_to(timeline.n2p(2011)).to_edge(
            UP, buff=-connection_img.height
        )

        self.add(connection_img)

        self.play(
            connection_img.animate.next_to(timeline.n2p(2011), direction=UP, buff=0.5),
            run_time=2,
        )
        connection_line = DashedLine(
            connection_img.get_bottom(), timeline.n2p(2011), color=WHITE
        )
        self.play(Create(connection_line))

        vincent_img.to_corner(UL, buff=0.75).shift(2 * RIGHT)
        vincent_label.next_to(vincent_img, direction=UP)

        self.play(
            LaggedStart(FadeIn(vincent_img), Write(vincent_label), lag_ratio=0.3),
            run_time=2,
        )

        self.wait(0.8)

        yang_img = ImageMobject("img/yang_song.jpg").scale_to_fit_width(1.5)
        yang_img.add(SurroundingRectangle(yang_img, color=WHITE, buff=0))
        yang_label = (
            Tex("Yang Song", color=WHITE).scale(0.7).next_to(yang_img, direction=UP)
        )
        generative_img = (
            ImageMobject("img/generative_modeling.png")
            .scale_to_fit_width(2)
            .set_z_index(2)
        )
        score_img = (
            ImageMobject("img/score_based.png").scale_to_fit_width(2).set_z_index(3)
        )

        generative_img.move_to(timeline.n2p(2019)).to_edge(
            UP, buff=-generative_img.height
        )
        score_img.move_to(timeline.n2p(2021)).to_edge(DOWN, buff=-score_img.height)

        self.add(generative_img, score_img)

        self.play(
            LaggedStart(
                generative_img.animate.next_to(
                    timeline.n2p(2019), direction=UP, buff=0.5
                ),
                score_img.animate.next_to(timeline.n2p(2021), direction=UP, buff=0.5),
                run_time=2,
            )
        )

        generative_line = DashedLine(
            generative_img.get_bottom(), timeline.n2p(2019), color=WHITE
        )
        score_line = DashedLine(score_img.get_bottom(), timeline.n2p(2021), color=WHITE)

        self.play(Create(generative_line), Create(score_line))

        yang_img.to_corner(UR, buff=0.75)
        yang_label.next_to(yang_img, direction=UP)

        self.play(
            LaggedStart(FadeIn(yang_img), Write(yang_label), lag_ratio=0.3),
            run_time=2,
        )

        self.wait(0.7)

        self.play(
            FadeOut(
                vincent_img,
                vincent_label,
                connection_img,
                connection_line,
                yang_img,
                yang_label,
                score_img,
                generative_img,
                generative_line,
                score_line,
            ),
            Uncreate(timeline),
        )

        # Hyvärinen"s approach
        self.next_section(skip_animations=False)

        score_matching_naive = MathTex(
            r"\mathbb{E}_{x \sim p(x)} \left[ \| ",
            r"s_{\theta}",
            r"(x) - ",
            r"\nabla_x \log p(x)",
            r"\|^2 \right]",
            color=WHITE,
        ).scale(0.8)
        score_matching_naive[1].set_color(TEAL)

        self.play(Write(score_matching_naive))

        self.wait(0.6)

        rect_score_gt = SurroundingRectangle(
            score_matching_naive[3], color=WHITE, buff=0.1
        )
        rect_score_gt_label = (
            Tex(r"Ground truth score", color=WHITE)
            .scale(0.6)
            .next_to(rect_score_gt, direction=DOWN, buff=0.5)
        )
        self.play(Create(rect_score_gt))
        self.play(Write(rect_score_gt_label))

        self.wait(0.7)

        score_hyvarinen = MathTex(
            r"\mathbb{E}_{x \sim p(x)} \left[",
            r"\text{tr}( \nabla",
            r"s_{\theta}",
            r"(x))",
            r"+ \frac{1}{2} \|",
            r"s_{\theta}",
            r"(x) \|^2 \right]",
            color=WHITE,
        ).scale(0.8)
        score_hyvarinen[2].set_color(TEAL)
        score_hyvarinen[5].set_color(TEAL)

        hyvarinen_img = (
            ImageMobject("img/hyvarinen.jpg")
            .scale_to_fit_width(1.5)
            .to_corner(UL, buff=0.75)
        )
        hyvarinen_img.add(SurroundingRectangle(hyvarinen_img, color=WHITE, buff=0))
        hyvarinen_label = (
            Tex("Aapo Hyvärinen", color=WHITE)
            .scale(0.7)
            .next_to(hyvarinen_img, direction=UP)
        )
        estimation_img = ImageMobject("img/estimation.png").scale_to_fit_width(2)
        estimation_img.to_corner(UR, buff=0.75).shift(3 * UP)

        self.play(FadeIn(hyvarinen_img), Write(hyvarinen_label))
        self.play(estimation_img.animate.shift(3 * DOWN))

        self.play(
            score_matching_naive.animate.next_to(score_hyvarinen, direction=UP, buff=2),
            FadeOut(rect_score_gt, rect_score_gt_label),
        )

        arrow = Arrow(
            score_matching_naive.get_bottom(),
            score_hyvarinen.get_top(),
            color=WHITE,
            buff=0.3,
        )

        self.play(Create(arrow))
        self.play(Write(score_hyvarinen))

        self.wait(0.7)

        self.wait(2)
        rect = SurroundingRectangle(score_hyvarinen, color=WHITE, buff=0.1)
        rect_label = (
            Tex(r"Only requires samples from $p(x)$", color=WHITE)
            .scale(0.6)
            .next_to(rect, direction=DOWN, buff=0.5)
        )

        self.play(Create(rect))
        self.play(Write(rect_label))

        self.wait(0.7)
        self.play(FadeOut(rect, rect_label))

        rect_jacobian = SurroundingRectangle(score_hyvarinen[1:4], color=RED, buff=0.1)
        jacobian_label = (
            Tex("Very costly in high dimensions!", color=RED)
            .scale(0.6)
            .next_to(rect_jacobian, direction=DOWN, buff=0.5)
        )

        self.play(Create(rect_jacobian))
        self.play(Write(jacobian_label))

        self.wait(0.7)

        # Denoising score matching
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

        score_hyvarinen_cp = score_hyvarinen.copy()
        VGroup(score_hyvarinen_cp, denoising_score_matching).arrange(
            direction=RIGHT, buff=1
        ).move_to(ORIGIN)

        denoising_score_label = (
            Tex("Denoising Score Matching", color=WHITE)
            .scale(0.7)
            .next_to(denoising_score_matching, direction=UP, buff=1)
        )

        self.play(
            FadeOut(hyvarinen_img, hyvarinen_label, estimation_img),
            FadeOut(score_matching_naive, arrow, rect_jacobian, jacobian_label),
            score_hyvarinen.animate.move_to(score_hyvarinen_cp),
        )

        hyvarinen_label = (
            Tex(r"Hyvärinen's method", color=WHITE)
            .scale(0.7)
            .next_to(score_hyvarinen, direction=UP, buff=1)
        )

        self.play(Write(denoising_score_matching))

        self.play(FadeIn(denoising_score_label, hyvarinen_label))

        self.wait(0.8)

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_6()
    scene.render()
