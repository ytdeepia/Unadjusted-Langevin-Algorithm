from manim import *


class Scene2_5(Scene):
    def construct(self):

        txt = Text("Thanks for watching!").scale(1.2).to_edge(UP, buff=1.0)

        self.play(Write(txt))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_5()
    scene.render()
