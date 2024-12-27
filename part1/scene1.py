from manim import *
import numpy as np


class Scene1_1(Scene):
    def construct(self):

        # Denoising an image
        self.next_section(skip_animations=False)

        encoder = Polygon(
            [-0.8, 1, 0], [0.8, 0.4, 0], [0.8, -0.4, 0], [-0.8, -1, 0], color=PURPLE
        )
        encoder_txt = MathTex(r"E").move_to(encoder.get_center())
        bottleneck = Rectangle(width=0.5, height=0.8, color=BLUE).next_to(
            encoder, direction=RIGHT, buff=0.0
        )
        decoder = Polygon(
            [-0.8, 0.4, 0], [0.8, 1, 0], [0.8, -1, 0], [-0.8, -0.4, 0], color=PURPLE
        ).next_to(bottleneck, direction=RIGHT, buff=0.0)
        decoder_txt = MathTex(r"D").move_to(decoder.get_center())
        autoencoder = VGroup(
            encoder, decoder, bottleneck, encoder_txt, decoder_txt
        ).move_to(ORIGIN)
        autoencoder_brace = Brace(autoencoder, UP)
        autoencoder_txt = autoencoder_brace.get_text("Autoencoder")

        img1 = ImageMobject("img/butterfly.png").scale_to_fit_width(2)
        img1.add(SurroundingRectangle(img1, color=WHITE, buff=0.0))
        img1_noisy = ImageMobject("img/butterfly_noisy.png").scale_to_fit_width(2)
        img1_noisy.add(SurroundingRectangle(img1_noisy, color=WHITE, buff=0.0))
        img1_noisy.next_to(encoder, direction=LEFT, buff=1)
        img1.next_to(decoder, direction=RIGHT, buff=1)

        arrowin = Arrow(start=img1_noisy.get_right(), end=encoder.get_left(), buff=0.1)
        arrowout = Arrow(start=decoder.get_right(), end=img1.get_left(), buff=0.1)

        self.play(FadeIn(autoencoder), Write(autoencoder_brace), Write(autoencoder_txt))
        self.play(LaggedStart(FadeIn(img1_noisy), GrowArrow(arrowin), lag_ratio=0.5))
        self.play(LaggedStart(GrowArrow(arrowout), FadeIn(img1), lag_ratio=0.5))

        self.wait(0.8)

        self.play(
            FadeOut(
                img1_noisy,
                arrowin,
                img1,
                arrowout,
                autoencoder_brace,
                autoencoder_txt,
            ),
            autoencoder.animate.scale(0.5).to_edge(UP),
        )

        autoencoder_brace = Brace(autoencoder, DOWN)
        autoencoder_txt = autoencoder_brace.get_tex(r"s_{\theta}").set_color(TEAL)
        self.play(FadeIn(autoencoder_brace), Write(autoencoder_txt))

        # Underlying distribution
        self.next_section(skip_animations=False)

        manifold = ParametricFunction(
            lambda t: np.array([t, 0.5 * np.sin(t) + 0.2 * t, 0]),
            t_range=np.array([-PI, PI]),
            color=BLUE_D,
            stroke_width=4,
            z_index=-1,
        ).shift(DOWN)
        manifold_txt = (
            Text("Image Manifold", color=BLUE_D)
            .next_to(manifold, direction=UL, buff=-1)
            .scale(0.5)
        )

        data_pt = Dot(color=ORANGE).move_to(0.5 * LEFT + 0.2 * UP)
        data_pt_txt = MathTex(r"\tilde{x}", color=ORANGE).next_to(
            data_pt, direction=UP, buff=0.15
        )

        proj_pt = Dot(color=BLUE).move_to(manifold.point_from_proportion(0.45))
        proj_pt_txt = MathTex(r"\hat{x}", color=BLUE).next_to(
            proj_pt, direction=DOWN, buff=0.15
        )

        arrow_proj = Arrow(
            data_pt.get_center(), proj_pt.get_center(), buff=0.1, color=WHITE
        )
        arrow_proj_txt = (
            MathTex(r"s_{\theta}", color=TEAL)
            .next_to(arrow_proj, direction=RIGHT, buff=0.1)
            .shift(0.2 * UP)
        )

        self.play(Create(manifold))
        self.play(Write(manifold_txt))

        self.play(
            LaggedStart(
                AnimationGroup(FadeIn(data_pt), Write(data_pt_txt)),
                AnimationGroup(GrowArrow(arrow_proj), Write(arrow_proj_txt)),
                AnimationGroup(FadeIn(proj_pt), Write(proj_pt_txt)),
                lag_ratio=0.8,
            ),
            run_time=2,
        )

        self.wait(0.8)

        # Tweedie formula
        self.next_section(skip_animations=False)

        tweedie = MathTex(
            r"s_{\theta}",
            "(x)",
            r"= x + ",
            r"\sigma^2",
            r" \nabla \log p_{",
            r"\sigma",
            r"}(x)",
            color=WHITE,
        ).scale(1.5)
        tweedie[0].set_color(TEAL)
        tweedie[3].set_color(ORANGE)
        tweedie[5].set_color(ORANGE)
        tweedie.to_edge(UP, buff=1).scale(0.8)
        tweedie_rect = SurroundingRectangle(tweedie, color=WHITE, buff=0.25)

        self.play(FadeOut(autoencoder, autoencoder_brace, autoencoder_txt))
        self.play(Create(tweedie_rect), Write(tweedie))

        self.wait(0.8)

        self.wait(0.4)

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN), run_time=0.7)

        txt = Text("Let's find out!").scale(0.8)

        self.play(
            FadeIn(txt),
            Flash(
                txt,
                color=YELLOW_D,
                line_length=1.2,
                num_lines=24,
                flash_radius=txt.width / 2 + 0.4,
            ),
            run_time=1.5,
        )

        self.play(Unwrite(txt))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_1()
    scene.render()
