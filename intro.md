# a formula a day

I made a [framework for JAX visualisation](https://github.com/aminehd/jaxvis) where you can annotate your JAX-compatible function with it, and after it runs it generates a graph with the output of each intermediate operation that JAX generated. There is also another visualizer, which visualizes the transformation that happens to the function's input when the function is applied. You can think about grad as one such transformation.

I think seeing these formulas can be very helpful in learning them, and the generated images can be another learning channel for multimodal models or for agents to "see" what equations look like.

On this website, I am gonna experiment with different functions each day and publish if they turned out alright. Btw, to generate the images, I just need to annotate them with `jaxvis.draw`.
