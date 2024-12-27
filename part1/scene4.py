from manim import *

import numpy as np
import matplotlib.pyplot as plt


class Scene1_4(Scene):
    def construct(self):
        # Tweedie's formula detailed
        self.next_section(skip_animations=False)

        tweedie_init = MathTex(
            r"\mathbb{E}[\mathbf{x} \mid ",
            r"\mathbf{y}",
            r"]",
            r"=",
            r"\mathbf{y}",
            r"+",
            r"\sigma^2",
            r"\nabla \log p(",
            r"\mathbf{y}",
            r")",
        )
        tweedie_init.font_size = 38
        tweedie_init[1].set_color(ORANGE)
        tweedie_init[4].set_color(ORANGE)
        tweedie_init[8].set_color(ORANGE)

        tweedie_txt = (
            Text("Tweedie's formula").scale(0.8).next_to(tweedie_init, UP, buff=1.5)
        )
        tweedie_txt_ul = Underline(tweedie_txt)

        self.play(Write(tweedie_init))
        self.play(Write(tweedie_txt), Create(tweedie_txt_ul))

        self.wait(0.7)

        rect = SurroundingRectangle(tweedie_init[0:3], color=WHITE, buff=0.1)
        self.wait(1.5)
        self.play(Create(rect))

        label = Tex("Posterior mean").scale(0.8).next_to(rect, DOWN, buff=0.5)
        self.play(Write(label))

        self.wait(0.5)

        rect2 = SurroundingRectangle(tweedie_init[7:], color=WHITE, buff=0.1)
        self.play(Transform(rect, rect2), FadeOut(label))

        label = Tex("Score").scale(0.8).next_to(rect2, DOWN, buff=0.5)
        self.play(Write(label))

        self.wait(0.5)

        self.play(FadeOut(rect, label), run_time=0.8)

        self.wait(0.8)

        # Derivations to extract the posterior mean step 1
        self.next_section(skip_animations=False)

        score_frac = MathTex(
            r"\nabla \log p(\mathbf{y})", r"=\frac{\nabla p(\mathbf{y})}{p(\mathbf{y})}"
        )
        score_frac.font_size = 38

        self.play(
            FadeOut(tweedie_txt, tweedie_txt_ul, tweedie_init[:7]),
            ReplacementTransform(tweedie_init[7:], score_frac[0]),
        )
        self.play(Write(score_frac[1:]))

        self.wait(0.6)

        self.play(score_frac.animate.to_edge(UP))

        marginal = MathTex(
            r"p(",
            r"\mathbf{y}",
            r")=\int p(",
            r"\mathbf{y}",
            r"|\mathbf{x})p(\mathbf{x})d\mathbf{x}",
        )
        marginal.font_size = 38
        marginal[1].set_color(ORANGE)
        marginal[3].set_color(ORANGE)

        self.play(Write(marginal))

        self.wait(0.6)

        marginal_grad = MathTex(
            r"\nabla p(",
            r"\mathbf{y}",
            r")= \nabla \int  p(",
            r"\mathbf{y}",
            r"|\mathbf{x})p(\mathbf{x})d\mathbf{x}",
        )
        marginal_grad.font_size = 38
        marginal_grad[1].set_color(ORANGE)
        marginal_grad[3].set_color(ORANGE)

        self.play(Transform(marginal, marginal_grad))

        self.wait(0.7)

        marginal_grad2 = MathTex(
            r"\nabla p(",
            r"\mathbf{y}",
            r")= \int \nabla p(",
            r"\mathbf{y}",
            r"|\mathbf{x})p(\mathbf{x})d\mathbf{x}",
        )
        marginal_grad2.font_size = 38
        marginal_grad2[1].set_color(ORANGE)
        marginal_grad2[3].set_color(ORANGE)

        self.wait(2)
        self.play(Transform(marginal, marginal_grad2))

        self.wait(0.6)

        self.play(marginal.animate.next_to(score_frac, DOWN, buff=0.5), run_time=0.7)

        rect = SurroundingRectangle(VGroup(marginal, score_frac), color=WHITE, buff=0.2)
        step1_txt = Text("Step 1: marginalize the score")
        step1_txt.font_size = 16
        step1_txt.to_corner(UL)

        self.play(Create(rect))
        step1 = VGroup(rect, marginal, score_frac)
        self.play(step1.animate.scale(0.5).next_to(step1_txt, DOWN, buff=0.2))
        self.play(Write(step1_txt))

        # Step 2
        self.next_section(skip_animations=False)

        likelihood = MathTex(
            r"p(",
            r"\mathbf{y}",
            r"\mid \mathbf{x}) \propto \exp \left( -\frac{1}{2 \sigma^2} \|",
            r"\mathbf{y}",
            r"- \mathbf{x} \|^2  \right)",
        )
        likelihood.font_size = 38
        likelihood[1].set_color(ORANGE)
        likelihood[3].set_color(ORANGE)

        likelihood_grad = MathTex(
            r"\nabla p(",
            r"\mathbf{y}",
            r"\mid \mathbf{x}) = -\frac{1}{\sigma^2}(",
            r"\mathbf{y}",
            r"- \mathbf{x})p(",
            r"\mathbf{y}",
            r"\mid \mathbf{x})",
        )
        likelihood_grad.font_size = 38
        likelihood_grad[1].set_color(ORANGE)
        likelihood_grad[3].set_color(ORANGE)
        likelihood_grad[5].set_color(ORANGE)

        marginal_with_likelihood = MathTex(
            r"\nabla p(",
            r"\mathbf{y}",
            r")=\int -\frac{1}{\sigma^2}(",
            r"\mathbf{y}",
            r"- \mathbf{x})p(",
            r"\mathbf{y}",
            r"\mid \mathbf{x})p(\mathbf{x})d\mathbf{x}",
        )
        marginal_with_likelihood.font_size = 38
        marginal_with_likelihood[1].set_color(ORANGE)
        marginal_with_likelihood[3].set_color(ORANGE)
        marginal_with_likelihood[5].set_color(ORANGE)

        self.play(Write(likelihood))

        self.wait(0.6)

        self.play(likelihood.animate.next_to(likelihood_grad, UP, buff=0.5))
        self.play(Write(likelihood_grad))

        self.play(
            VGroup(likelihood, likelihood_grad).animate.next_to(
                marginal_with_likelihood, UP, buff=0.5
            ),
            run_time=0.8,
        )

        self.play(Write(marginal_with_likelihood))

        self.wait(0.7)

        rect2 = SurroundingRectangle(
            VGroup(likelihood, likelihood_grad, marginal_with_likelihood),
            color=WHITE,
            buff=0.2,
        )
        step2 = VGroup(rect2, likelihood, likelihood_grad, marginal_with_likelihood)

        self.play(Create(rect2))

        step2_txt = Text("Step 2: Gaussian likelihood")
        step2_txt.font_size = 16
        step2_txt.to_corner(UR)

        self.play(step2.animate.scale(0.5).next_to(step2_txt, DOWN, buff=0.2))

        self.play(Write(step2_txt))

        self.wait(0.8)
        # Final step
        self.next_section(skip_animations=False)

        integral_formula = MathTex(
            r"\nabla p(",
            r"\mathbf{y}",
            r")=",
            r"-\frac{1}{\sigma^2}",
            r"\mathbf{y}",
            r"\int p(",
            r"\mathbf{y}",
            r"\mid \mathbf{x})p(\mathbf{x})d\mathbf{x}",
            r"+",
            r"\frac{1}{\sigma^2}",
            r"\int p(",
            r"\mathbf{y}",
            r"\mid \mathbf{x})\mathbf{x}p(\mathbf{x})d\mathbf{x}",
        )

        integral_formula.font_size = 32
        integral_formula[1].set_color(ORANGE)
        integral_formula[4].set_color(ORANGE)
        integral_formula[6].set_color(ORANGE)
        integral_formula[11].set_color(ORANGE)

        self.play(
            Write(integral_formula),
        )

        self.wait(0.7)

        brace_y = Brace(integral_formula[4:8], UP, buff=0.1)
        brace_y_text = brace_y.get_tex(r"\mathbf{y}", r"p(", r"\mathbf{y}", r")").scale(
            0.8
        )
        brace_y_text[0].set_color(ORANGE)
        brace_y_text[2].set_color(ORANGE)

        brace_posterior_mean = Brace(integral_formula[10:], DOWN, buff=0.1)
        brace_posterior_mean_text = brace_posterior_mean.get_tex(
            r"\mathbb{E}[\mathbf{x} \mid", r"\mathbf{y}", r"]p(", r"\mathbf{y}", r")"
        ).scale(0.8)
        brace_posterior_mean_text[1].set_color(ORANGE)
        brace_posterior_mean_text[3].set_color(ORANGE)

        self.wait(1)
        self.play(Create(brace_y), Write(brace_y_text))

        self.wait(0.8)

        self.play(Create(brace_posterior_mean), Write(brace_posterior_mean_text))

        self.wait(0.8)

        self.wait(1)
        tweedie_formula = MathTex(
            r"\nabla \log p(",
            r"\mathbf{y}",
            r")=",
            r"-\frac{1}{\sigma^2}",
            r"\left(",
            r"\mathbf{y}",
            r"-",
            r"\mathbb{E}[\mathbf{x} \mid",
            r"\mathbf{y}",
            r"]",
            r"\right)",
        )
        tweedie_formula.font_size = 32
        tweedie_formula[1].set_color(ORANGE)
        tweedie_formula[5].set_color(ORANGE)
        tweedie_formula[8].set_color(ORANGE)

        self.play(
            FadeOut(
                brace_y,
                brace_y_text,
                brace_posterior_mean,
                brace_posterior_mean_text,
            ),
            Transform(integral_formula, tweedie_formula),
        )

        self.wait(0.6)

        self.play(FadeOut(*self.mobjects))

        txt = Text("Time for some animations!").shift(2.5 * UP)
        txt.font_size = 28
        txt_ul = Underline(txt)
        self.play(LaggedStart(Write(txt), Create(txt_ul), lag_ratio=0.2), run_time=1.5)

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_4()
    scene.render()
