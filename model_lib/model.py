import json
import os

from flask_psc_model import ModelBase, load_metadata

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class ModelName(ModelBase):

    #: load the `model.yaml` metadata file from up the filesystem hierarchy;
    #: this will be used to avoid hardcoding the below filenames in this file
    metadata = load_metadata(__file__)

    #: a list of input filenames; specifying the `input_filenames` attribute is required to configure the model app
    input_filenames = list(metadata.inputs)

    #: a list of output filenames; specifying the `output_filenames` attribute is required to configure the model app
    output_filenames = list(metadata.outputs)

    def __init__(self):
        """Load the model files and do any initialization.

        A single instance of this model class will be reused multiple times to perform inference
        on multiple input files so any slow initialization steps such as reading in a data
        files or loading an inference graph to GPU should be done here.

        This function should require no arguments, or provide appropriate defaults for all arguments.

        NOTE: The `__init__` function and `run` function may not be called from the same thread so extra
        care may be needed if using frameworks such as Tensorflow that make use of thread locals.
        """

    def run(self, input_path, output_path):
        """Run the model on the given input file paths and write to the given output file paths.

        The input files paths followed by the output file paths will be passed into this function as
        positional arguments in the same order as specified in `input_filenames` and `output_filenames`.

        For example:
        ```
        class SingleInputOutputModel(ModelBase):
            input_filenames = ['input.txt']
            output_filenames = ['output.json']

            def run(self, input, output):
                run_the_model(input, output)

        class MultipleInputOutputModel(ModelBase):
            input_filenames = ['input1.png', 'input2.json', 'input3.txt']
            output_filenames = ['output1.png', 'output2.json']

            def run(self, input1, input2, input3, output1, output2):
                run_the_model(input1, input2, input3, output1, output2)
        ```
        """

if __name__ == '__main__':
    # run the model independently from the full application; can be useful for testing
    #
    # to run from the repository root:
    #     python -m model_lib.model /path/to/input.txt /path/to/output.json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the input data filepath')
    parser.add_argument('output', help='the output results filepath')
    args = parser.parse_args()

    model = ModelName()
    model.run(args.input, args.output)
