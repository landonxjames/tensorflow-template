# TensorFlow Template

## Architecture
1. `main.py`
- input: either config file (through cli) or config object (calling within python)
- output: trained model or test results
- three modes: train, test, forward
- train saves and updates the model; test evaluates outputs, and forward just gives outputs.

2. `cli.py`
- contains all configs, configurable via cli. Calls the `main.py` module's `main` function.

3. `read_data.py`
- called by main, with `config` containing data root directory, 

4. `model.py`
- model consists of four components: forward, loss, and grads, update
- forward: function that models the distribution of interest. 
Training and test setup can be different (e.g. dropout).
- loss: computes the loss between the tower's output and labels
- grads: the gradients of the loss. In TensorFlow, this is a simple: `compute_gradients`.
Some post-processing might be applied (e.g. grad clipping).
- update: update the weights of the tower. Typically, this is just scaled grads.
- Training involves all steps above; testing only involves up to loss. 
In real time, only forward.
- In multi-GPU case, the grads of the towers are averaged and used for the update.
- this is all controlled in main.
- other parts are: `save` function, `log` function, 

5. `evaluator.py`
- Often times evaluation is not straightforward within the model, such as precision / recall.
This module uses the output of the model's forward pass 
to output the quantitative analysis of the output.
- Often times evaluation program is provided by the authors of the dataset.
In such case, this acts as the interface between the program and the current framework.

6. `trainer`
- I don't know