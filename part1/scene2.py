from manim import *


class Scene1_2(Scene):
    def construct(self):
        # Training an autoencoder
        self.next_section(skip_animations=False)

        img_x = ImageMobject("img/butterfly.png").scale_to_fit_width(2)
        img_x.add(SurroundingRectangle(img_x, color=WHITE, buff=0.0))
        img_xhat = ImageMobject("img/butterfly_denoised.png").scale_to_fit_width(2)
        img_xhat.add(SurroundingRectangle(img_xhat, color=WHITE, buff=0.0))
        img_y = ImageMobject("img/butterfly_noisy.png").scale_to_fit_width(2)
        img_y.add(SurroundingRectangle(img_y, color=WHITE, buff=0.0))

        x_label = MathTex(r"\mathbf{x}")
        xhat_label = MathTex(r"\hat{\mathbf{x}}", color=TEAL)
        y_label = MathTex(r"\mathbf{y}", color=ORANGE)

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
        autoencoder = (
            VGroup(encoder, decoder, bottleneck, encoder_txt, decoder_txt)
            .move_to(ORIGIN)
            .scale(0.8)
        )
        autoencoder_brace = Brace(autoencoder, UP)
        autoencoder_txt = autoencoder_brace.get_text("Denoiser")

        img_y.next_to(encoder, direction=LEFT, buff=1)
        y_label.next_to(img_y, direction=UP, buff=0.25)
        img_xhat.next_to(decoder, direction=RIGHT, buff=1)
        xhat_label.next_to(img_xhat, direction=UP, buff=0.25)
        img_x.next_to(img_xhat, direction=RIGHT, buff=0.5)
        x_label.next_to(img_x, direction=UP, buff=0.25)

        arrowin = Arrow(start=img_y.get_right(), end=encoder.get_left(), buff=0.1)
        arrowout = Arrow(start=decoder.get_right(), end=img_xhat.get_left(), buff=0.1)
        Group(
            autoencoder,
            autoencoder_brace,
            autoencoder_txt,
            img_y,
            img_x,
            img_xhat,
            xhat_label,
            x_label,
            y_label,
            arrowin,
            arrowout,
        ).move_to(ORIGIN)

        img_y_cp = img_y.copy()
        y_label_cp = y_label.copy()

        Group(img_y_cp, y_label_cp).move_to(ORIGIN)

        self.play(FadeIn(img_y_cp, y_label_cp))

        self.wait(0.8)

        self.play(img_y_cp.animate.move_to(img_y), y_label_cp.animate.move_to(y_label))

        self.play(
            LaggedStart(
                GrowArrow(arrowin),
                AnimationGroup(
                    Create(autoencoder_txt), FadeIn(autoencoder, autoencoder_brace)
                ),
                lag_ratio=0.6,
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                AnimationGroup(FadeIn(img_xhat), FadeIn(xhat_label)),
                lag_ratio=0.5,
            )
        )

        self.wait(0.8)

        self.play(FadeIn(img_x, x_label))

        self.wait(0.6)

        self.wait(0.8)

        self.wait(1.5)
        loss = MathTex(
            r"\mathcal{L} = " r"\|",
            r"\hat{\mathbf{x}}",
            r"- \mathbf{x}",
            r"\|^2",
        )
        loss.font_size = 38
        loss[1].set_color(TEAL)

        self.play(
            LaggedStart(
                Group(
                    img_xhat,
                    img_x,
                    img_y_cp,
                    autoencoder,
                    autoencoder_brace,
                    autoencoder_txt,
                    arrowin,
                    arrowout,
                ).animate.to_edge(UP),
                ReplacementTransform(VGroup(x_label, xhat_label, y_label_cp), loss),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.wait(2)
        total_loss = MathTex(
            r"\mathcal{L}_{total} = \frac{1}{N} \sum_{i=1}^N \|",
            r"\hat{\mathbf{x}}_i",
            r"- \mathbf{x}_i",
            r"\|^2",
        )
        total_loss.font_size = 38
        total_loss[1].set_color(TEAL)

        self.play(ReplacementTransform(loss, total_loss))

        loss_txt = (
            Text("Loss function").next_to(total_loss, direction=UP, buff=1).scale(0.5)
        )
        loss_txt_ul = Underline(loss_txt)

        self.play(
            LaggedStart(
                FadeOut(
                    img_xhat,
                    img_x,
                    img_y_cp,
                    autoencoder,
                    autoencoder_brace,
                    autoencoder_txt,
                    arrowin,
                    arrowout,
                ),
                AnimationGroup(Write(loss_txt), GrowFromEdge(loss_txt_ul, LEFT)),
                lag_ratio=0.8,
            ),
            run_time=2.5,
        )

        self.wait(0.7)

        self.wait(3)
        l2_exp = MathTex(
            r"\arg \min_{",
            r"\hat{\mathbf{x}}}",
            r" \mathbb{E} \left[ \|",
            r"\hat{\mathbf{x}}",
            r"- ",
            r"\mathbf{x}",
            r"\|^2 \mid ",
            r"\mathbf{y}",
            r" \right]",
        )
        l2_exp.font_size = 38

        l2_exp[1].set_color(TEAL)
        l2_exp[3].set_color(TEAL)
        l2_exp[7].set_color(ORANGE)

        self.play(
            ReplacementTransform(total_loss, l2_exp),
        )

        self.wait(0.7)

        # MMSE to posterior mean derivations
        self.next_section(skip_animations=False)

        self.wait(0.8)

        txt = Text("What is the denoiser learning?").scale(0.6).to_edge(UP, buff=1.5)

        self.wait(1.5)
        self.play(Write(txt), FadeOut(loss_txt, loss_txt_ul))

        self.wait(0.6)

        txt2 = (
            Text("The trained denoiser approximates the posterior mean")
            .scale(0.6)
            .to_edge(UP, buff=1.5)
        )
        self.wait(2)
        self.play(ReplacementTransform(txt, txt2))

        self.wait(0.7)

        self.play(FadeOut(txt2))
        exp_0 = MathTex(
            r"\mathbb{E} [ \|",
            r"\hat{\mathbf{x}}",
            r" - \mathbf{x}\|^2 \mid",
            r"\mathbf{y}",
            r"] = \mathbb{E} [ (",
            r"\hat{\mathbf{x}}",
            r"- \mathbf{x})^\top(",
            r"\hat{\mathbf{x}}",
            r"- \mathbf{x}) \mid",
            r"\mathbf{y}",
            r"]",
        )
        exp_0.font_size = 38
        exp_0[1].set_color(TEAL)
        exp_0[3].set_color(ORANGE)
        exp_0[5].set_color(TEAL)
        exp_0[7].set_color(TEAL)
        exp_0[9].set_color(ORANGE)

        self.play(ReplacementTransform(l2_exp, exp_0))

        self.wait(0.8)
        exp_1 = MathTex(
            r"\mathbb{E} [ \|",
            r"\hat{\mathbf{x}}",
            r"- \mathbf{x}\|^2 \mid",
            r"\mathbf{y}",
            r"] = \|",
            r"\hat{\mathbf{x}}",
            r"\|^2 + - 2",
            r"\hat{\mathbf{x}}",
            r"^\top \mathbb{E}[\mathbf{x} \mid",
            r"\mathbf{y}",
            r"] + \mathbb{E} [ \|\mathbf{x}\|^2 \mid ",
            r"\mathbf{y}",
            r"]",
        )
        exp_1.font_size = 38

        exp_1[1].set_color(TEAL)
        exp_1[3].set_color(ORANGE)
        exp_1[5].set_color(TEAL)
        exp_1[7].set_color(TEAL)
        exp_1[9].set_color(ORANGE)
        exp_1[11].set_color(ORANGE)

        self.play(exp_0.animate.shift(UP))
        self.play(Write(exp_1), run_time=2)

        self.wait(0.8)

        optimality_condition = MathTex(
            r"\frac{\partial}{\partial",
            r"\hat{\mathbf{x}}",
            r"} \mathbb{E} [ \|",
            r"\hat{\mathbf{x}}",
            r"- \mathbf{x}\|^2 \mid",
            r"\mathbf{y}",
            r"] = 0",
        )
        optimality_condition.font_size = 38
        optimality_condition[1].set_color(TEAL)
        optimality_condition[3].set_color(TEAL)
        optimality_condition[5].set_color(ORANGE)

        self.play(VGroup(exp_0, exp_1).animate.shift(UP))

        self.play(Write(optimality_condition))

        self.wait(0.6)

        optimality_condition_2 = MathTex(
            r"\frac{\partial}{\partial",
            r"\hat{\mathbf{x}}",
            r"}",
            r"\|",
            r"\hat{\mathbf{x}}",
            r"\|^2 - 2",
            r"\hat{\mathbf{x}}",
            r"^\top \mathbb{E}[\mathbf{x} \mid",
            r"\mathbf{y}",
            r"] + ",
            r"\mathbb{E} [ \|\mathbf{x}\|^2 \mid ",
            r"\mathbf{y}",
            r"]",
            r"= 0",
        )
        optimality_condition_2.font_size = 38
        optimality_condition_2[1].set_color(TEAL)
        optimality_condition_2[4].set_color(TEAL)
        optimality_condition_2[6].set_color(TEAL)
        optimality_condition_2[8].set_color(ORANGE)
        optimality_condition_2[11].set_color(ORANGE)

        self.play(
            LaggedStart(
                FadeOut(VGroup(exp_0, exp_1)),
                optimality_condition.animate.to_edge(UP),
            )
        )
        self.play(Write(optimality_condition_2))

        cross = Cross(
            VGroup(
                optimality_condition_2[10],
                optimality_condition_2[11],
                optimality_condition_2[12],
            )
        )
        self.wait(2)
        self.play(Write(cross))
        self.wait(1)
        self.play(
            FadeOut(
                VGroup(
                    optimality_condition_2[10],
                    optimality_condition_2[11],
                    optimality_condition_2[12],
                    cross,
                ),
                shift=0.5 * DOWN,
            )
        )

        self.wait(0.8)

        optimality_condition_2.remove(
            optimality_condition_2[10],
            optimality_condition_2[11],
            optimality_condition_2[12],
        )
        optimality_condition_3 = MathTex(
            r"2",
            r"\hat{\mathbf{x}}",
            r"- 2",
            r"\mathbb{E}[\mathbf{x} \mid",
            r"\mathbf{y}",
            r"] = 0",
        )

        optimality_condition_3.font_size = 38
        optimality_condition_3[1].set_color(TEAL)
        optimality_condition_3[4].set_color(ORANGE)

        self.play(ReplacementTransform(optimality_condition_2, optimality_condition_3))

        self.wait(0.7)

        final_expression = MathTex(
            r"\hat{\mathbf{x}}",
            r" = \mathbb{E}[\mathbf{x} \mid",
            r"\mathbf{y}",
            r"]",
        )
        final_expression.font_size = 38

        final_expression[0].set_color(TEAL)
        final_expression[2].set_color(ORANGE)

        self.play(optimality_condition_3.animate.shift(1.5 * UP))
        self.play(Write(final_expression))

        rect = SurroundingRectangle(final_expression, color=WHITE, buff=0.2)

        self.play(
            LaggedStart(
                FadeOut(optimality_condition_3, optimality_condition), Create(rect)
            )
        )
        self.wait(0.8)

        txt_mmse = (
            Text("The MMSE estimator is the posterior mean for Gaussian denoising")
            .next_to(rect, direction=UP, buff=1.5)
            .scale(0.5)
        )

        self.play(Write(txt_mmse), run_time=2)

        txt_mmse_ul = Underline(txt_mmse)

        self.play(ShowPassingFlash(txt_mmse_ul), run_time=1)

        self.wait(0.7)

        self.play(FadeOut(VGroup(rect, final_expression)))

        self.play(FadeOut(txt_mmse, shift=0.5 * RIGHT))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_2()
    scene.render()
